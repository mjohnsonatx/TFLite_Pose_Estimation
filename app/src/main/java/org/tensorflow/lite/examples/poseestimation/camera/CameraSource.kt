/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================
*/

package org.tensorflow.lite.examples.poseestimation.camera

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Bitmap
import android.graphics.ImageFormat
import android.graphics.Rect
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.media.Image
import android.media.ImageReader
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import android.view.Surface
import android.view.SurfaceView
import kotlinx.coroutines.suspendCancellableCoroutine
import org.tensorflow.lite.examples.poseestimation.VisualizationUtils
import org.tensorflow.lite.examples.poseestimation.data.Person
import org.tensorflow.lite.examples.poseestimation.imaging.Rotation
import org.tensorflow.lite.examples.poseestimation.imaging.YuvToRgbConverter
import org.tensorflow.lite.examples.poseestimation.ml.MoveNetMultiPose
import org.tensorflow.lite.examples.poseestimation.ml.PoseClassifier
import org.tensorflow.lite.examples.poseestimation.ml.PoseDetector
import org.tensorflow.lite.examples.poseestimation.ml.TrackerType
import java.util.Timer
import java.util.TimerTask
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException

class CameraSource(
    private val surfaceView: SurfaceView,
    private val listener: CameraSourceListener? = null
) {

    companion object {
        private const val PREVIEW_WIDTH = 640
        private const val PREVIEW_HEIGHT = 480

        /** Threshold for confidence score. */
        private const val MIN_CONFIDENCE = .2f
        private const val TAG = "Camera Source"

        /**
         * The camera sensor is landscape while the preview is portrait, so every frame is turned a
         * quarter turn clockwise. The rotation is folded into the YUV conversion, which means no
         * intermediate bitmap is created for it.
         */
        private val PREVIEW_ROTATION = Rotation.ROTATION_90

        /**
         * One image in flight, one queued and one being filled by the producer. Anything less makes
         * `acquireLatestImage` starve as soon as inference is slower than the capture interval.
         */
        private const val IMAGE_BUFFER_SIZE = 3

        /** How long teardown waits for the in-flight frame before giving up. */
        private const val SHUTDOWN_TIMEOUT_MILLIS = 500L
    }

    private val lock = Any()
    private var detector: PoseDetector? = null
    private var classifier: PoseClassifier? = null
    private var isTrackerEnabled = false

    /** Owns the reusable ARGB scratch array and the reusable preview bitmap. */
    private var yuvConverter: YuvToRgbConverter? = null

    /** Latest-frame-wins backpressure between the camera thread and inference. */
    private var frameProcessor: LatestFrameProcessor<Image>? = null
    private var inferenceExecutor: ExecutorService? = null

    /** Reused every frame so drawing to the surface does not allocate. */
    private val sourceRect = Rect()
    private val destinationRect = Rect()

    /** Frame count that have been processed so far in an one second interval to calculate FPS. */
    private var fpsTimer: Timer? = null

    // Written on the inference thread and read on the timer thread.
    @Volatile
    private var frameProcessedInOneSecondInterval = 0

    @Volatile
    private var framesPerSecond = 0

    /** Detects, characterizes, and connects to a CameraDevice (used for all camera operations) */
    private val cameraManager: CameraManager by lazy {
        val context = surfaceView.context
        context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
    }

    /** Readers used as buffers for camera still shots */
    private var imageReader: ImageReader? = null

    /** The [CameraDevice] that will be opened in this fragment */
    private var camera: CameraDevice? = null

    /** Internal reference to the ongoing [CameraCaptureSession] configured with our parameters */
    private var session: CameraCaptureSession? = null

    /** [HandlerThread] where all buffer reading operations run */
    private var imageReaderThread: HandlerThread? = null

    /** [Handler] corresponding to [imageReaderThread] */
    private var imageReaderHandler: Handler? = null
    private var cameraId: String = ""

    suspend fun initCamera() {
        camera = openCamera(cameraManager, cameraId)
        imageReader = ImageReader.newInstance(
            PREVIEW_WIDTH, PREVIEW_HEIGHT, ImageFormat.YUV_420_888, IMAGE_BUFFER_SIZE
        )
        imageReader?.setOnImageAvailableListener({ reader ->
            // acquireLatestImage already discards the backlog inside the reader; the processor
            // applies the same policy to the frame waiting on inference.
            val image = try {
                reader.acquireLatestImage()
            } catch (error: IllegalStateException) {
                Log.w(TAG, "Unable to acquire an image", error)
                null
            } ?: return@setOnImageAvailableListener

            val processor = frameProcessor
            if (processor == null) {
                // Nothing is running, so this frame must not be leaked back to the reader.
                image.close()
                return@setOnImageAvailableListener
            }
            processor.submit(image)
        }, imageReaderHandler)

        imageReader?.surface?.let { surface ->
            session = createSession(listOf(surface))
            val cameraRequest = camera?.createCaptureRequest(
                CameraDevice.TEMPLATE_PREVIEW
            )?.apply {
                addTarget(surface)
            }
            cameraRequest?.build()?.let {
                session?.setRepeatingRequest(it, null, null)
            }
        }
    }

    private suspend fun createSession(targets: List<Surface>): CameraCaptureSession =
        suspendCancellableCoroutine { cont ->
            camera?.createCaptureSession(targets, object : CameraCaptureSession.StateCallback() {
                override fun onConfigured(captureSession: CameraCaptureSession) =
                    cont.resume(captureSession)

                override fun onConfigureFailed(session: CameraCaptureSession) {
                    cont.resumeWithException(Exception("Session error"))
                }
            }, null)
        }

    @SuppressLint("MissingPermission")
    private suspend fun openCamera(manager: CameraManager, cameraId: String): CameraDevice =
        suspendCancellableCoroutine { cont ->
            manager.openCamera(cameraId, object : CameraDevice.StateCallback() {
                override fun onOpened(camera: CameraDevice) = cont.resume(camera)

                override fun onDisconnected(camera: CameraDevice) {
                    camera.close()
                }

                override fun onError(camera: CameraDevice, error: Int) {
                    if (cont.isActive) cont.resumeWithException(Exception("Camera error"))
                }
            }, imageReaderHandler)
        }

    fun prepareCamera() {
        for (cameraId in cameraManager.cameraIdList) {
            val characteristics = cameraManager.getCameraCharacteristics(cameraId)

            // We don't use a front facing camera in this sample.
            val cameraDirection = characteristics.get(CameraCharacteristics.LENS_FACING)
            if (cameraDirection != null &&
                cameraDirection == CameraCharacteristics.LENS_FACING_FRONT
            ) {
                continue
            }
            this.cameraId = cameraId
        }
    }

    fun setDetector(detector: PoseDetector) {
        synchronized(lock) {
            if (this.detector != null) {
                this.detector?.close()
                this.detector = null
            }
            this.detector = detector
        }
    }

    fun setClassifier(classifier: PoseClassifier?) {
        synchronized(lock) {
            if (this.classifier != null) {
                this.classifier?.close()
                this.classifier = null
            }
            this.classifier = classifier
        }
    }

    /**
     * Set Tracker for Movenet MuiltiPose model.
     */
    fun setTracker(trackerType: TrackerType) {
        isTrackerEnabled = trackerType != TrackerType.OFF
        (this.detector as? MoveNetMultiPose)?.setTracker(trackerType)
    }

    fun resume() {
        // Resuming twice would orphan the previous thread, executor and reusable bitmap.
        if (frameProcessor != null) return
        imageReaderThread = HandlerThread("imageReaderThread").apply { start() }
        imageReaderHandler = Handler(imageReaderThread!!.looper)
        yuvConverter = YuvToRgbConverter()
        val executor = Executors.newSingleThreadExecutor { runnable ->
            Thread(runnable, "poseInference").apply { isDaemon = true }
        }
        inferenceExecutor = executor
        frameProcessor = LatestFrameProcessor(
            executor = executor,
            recycle = { image -> closeQuietly(image) },
            onError = { error -> Log.e(TAG, "Frame processing failed", error) },
            process = { image -> processImage(image) }
        )
        fpsTimer = Timer()
        fpsTimer?.schedule(
            object : TimerTask() {
                override fun run() {
                    framesPerSecond = frameProcessedInOneSecondInterval
                    frameProcessedInOneSecondInterval = 0
                }
            },
            0,
            1000
        )
    }

    /**
     * Tears the pipeline down.
     *
     * Order matters: the camera stops producing first, then the queue is drained so every
     * outstanding [Image] is closed, and only afterwards are the reusable bitmap, pixel array and
     * the interpreters released. Doing it the other way round would recycle memory that the
     * in-flight inference is still reading.
     */
    fun close() {
        session?.close()
        session = null
        camera?.close()
        camera = null
        imageReader?.setOnImageAvailableListener(null, null)
        imageReader?.close()
        imageReader = null

        frameProcessor?.let { processor ->
            processor.close()
            if (!processor.awaitIdle(SHUTDOWN_TIMEOUT_MILLIS)) {
                Log.w(TAG, "Timed out waiting for the in-flight frame to finish")
            }
        }
        frameProcessor = null
        inferenceExecutor?.let { executor ->
            executor.shutdown()
            try {
                if (!executor.awaitTermination(SHUTDOWN_TIMEOUT_MILLIS, TimeUnit.MILLISECONDS)) {
                    executor.shutdownNow()
                }
            } catch (interrupted: InterruptedException) {
                executor.shutdownNow()
                Thread.currentThread().interrupt()
            }
        }
        inferenceExecutor = null

        stopImageReaderThread()

        yuvConverter?.close()
        yuvConverter = null

        synchronized(lock) {
            detector?.close()
            detector = null
            classifier?.close()
            classifier = null
        }

        fpsTimer?.cancel()
        fpsTimer = null
        frameProcessedInOneSecondInterval = 0
        framesPerSecond = 0
    }

    /**
     * Converts, analyses and renders one frame.
     *
     * The whole body runs while the converter's reusable bitmap is checked out, so the next camera
     * frame can never overwrite the pixels the interpreter is reading.
     */
    private fun processImage(image: Image) {
        val converter = yuvConverter ?: return
        converter.withConvertedFrame(image, PREVIEW_ROTATION) { bitmap ->
            analyse(bitmap)
        }
    }

    private fun analyse(bitmap: Bitmap) {
        val persons = mutableListOf<Person>()
        var classificationResult: List<Pair<String, Float>>? = null

        synchronized(lock) {
            detector?.estimatePoses(bitmap)?.let {
                persons.addAll(it)

                // if the model only returns one item, allow running the Pose classifier.
                if (persons.isNotEmpty()) {
                    classifier?.run {
                        classificationResult = classify(persons[0])
                    }
                }
            }
        }
        frameProcessedInOneSecondInterval++
        if (frameProcessedInOneSecondInterval == 1) {
            // send fps to view
            listener?.onFPSListener(framesPerSecond)
        }

        // if the model returns only one item, show that item's score.
        if (persons.isNotEmpty()) {
            listener?.onDetectedInfo(persons[0].score, classificationResult)
        }
        visualize(persons, bitmap)
    }

    /**
     * Draws the skeletons onto the reusable frame and blits it to the surface.
     *
     * The overlay is painted straight onto the converted frame; the previous implementation copied
     * the whole bitmap for every frame just to draw a handful of lines on it.
     */
    private fun visualize(persons: List<Person>, bitmap: Bitmap) {
        VisualizationUtils.drawBodyKeypointsInPlace(
            bitmap,
            persons.filter { it.score > MIN_CONFIDENCE },
            isTrackerEnabled
        )

        val holder = surfaceView.holder
        val surfaceCanvas = holder.lockCanvas() ?: return
        try {
            val screenWidth: Int
            val screenHeight: Int
            val left: Int
            val top: Int

            if (surfaceCanvas.height > surfaceCanvas.width) {
                val ratio = bitmap.height.toFloat() / bitmap.width
                screenWidth = surfaceCanvas.width
                left = 0
                screenHeight = (surfaceCanvas.width * ratio).toInt()
                top = (surfaceCanvas.height - screenHeight) / 2
            } else {
                val ratio = bitmap.width.toFloat() / bitmap.height
                screenHeight = surfaceCanvas.height
                top = 0
                screenWidth = (surfaceCanvas.height * ratio).toInt()
                left = (surfaceCanvas.width - screenWidth) / 2
            }

            sourceRect.set(0, 0, bitmap.width, bitmap.height)
            destinationRect.set(left, top, left + screenWidth, top + screenHeight)
            surfaceCanvas.drawBitmap(bitmap, sourceRect, destinationRect, null)
        } finally {
            holder.unlockCanvasAndPost(surfaceCanvas)
        }
    }

    private fun closeQuietly(image: Image) {
        try {
            image.close()
        } catch (error: RuntimeException) {
            Log.w(TAG, "Failed to release a camera frame", error)
        }
    }

    private fun stopImageReaderThread() {
        imageReaderThread?.quitSafely()
        try {
            imageReaderThread?.join()
            imageReaderThread = null
            imageReaderHandler = null
        } catch (e: InterruptedException) {
            Log.d(TAG, e.message.toString())
        }
    }

    interface CameraSourceListener {
        fun onFPSListener(fps: Int)

        fun onDetectedInfo(personScore: Float?, poseLabels: List<Pair<String, Float>>?)
    }
}
