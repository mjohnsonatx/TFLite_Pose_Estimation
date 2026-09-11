package org.tensorflow.lite.examples.poseestimation.camera

import android.graphics.Bitmap
import androidx.core.graphics.createBitmap

/**
 * Manages reusable frame buffers and bitmaps for the camera and inference pipeline.
 * Ensures allocations are not repeated on every frame for fixed-size camera inputs,
 * and guards against concurrent resource access while in an active inference path.
 */
class FrameResourceManager : AutoCloseable {

    private val lock = Any()
    private var inUse: Boolean = false
    private var isReleased: Boolean = false

    var currentWidth: Int = 0
        private set
    var currentHeight: Int = 0
        private set
    var currentRotation: Int = 0
        private set

    var rotatedBitmap: Bitmap? = null
        private set
    var visualizationBitmap: Bitmap? = null
        private set

    /**
     * Attempts to acquire exclusive access to pipeline resources for an inference run.
     * @return true if acquired, false if resources are currently in use or released.
     */
    fun acquire(): Boolean {
        synchronized(lock) {
            if (isReleased || inUse) {
                return false
            }
            inUse = true
            return true
        }
    }

    /**
     * Releases exclusive access to pipeline resources after an inference run completes.
     */
    fun release() {
        synchronized(lock) {
            inUse = false
        }
    }

    /**
     * Prepares and reallocates frame bitmaps only if dimensions or rotation changed.
     * Must be called while holding exclusive access or during initialization.
     */
    fun prepare(srcWidth: Int, srcHeight: Int, rotationDegrees: Int) {
        synchronized(lock) {
            if (isReleased) return

            val normalizedRotation = ((rotationDegrees % 360) + 360) % 360
            val targetWidth = if (normalizedRotation == 90 || normalizedRotation == 270) srcHeight else srcWidth
            val targetHeight = if (normalizedRotation == 90 || normalizedRotation == 270) srcWidth else srcHeight

            if (currentWidth != targetWidth || currentHeight != targetHeight || currentRotation != normalizedRotation || rotatedBitmap == null) {
                rotatedBitmap?.recycle()
                visualizationBitmap?.recycle()

                currentWidth = targetWidth
                currentHeight = targetHeight
                currentRotation = normalizedRotation

                rotatedBitmap = createBitmap(targetWidth, targetHeight)
                visualizationBitmap = createBitmap(targetWidth, targetHeight)
            }
        }
    }

    fun isInUse(): Boolean = synchronized(lock) { inUse }

    fun isReleased(): Boolean = synchronized(lock) { isReleased }

    /**
     * Recycles and clears all held resources.
     */
    override fun close() {
        synchronized(lock) {
            isReleased = true
            inUse = false
            rotatedBitmap?.recycle()
            rotatedBitmap = null
            visualizationBitmap?.recycle()
            visualizationBitmap = null
            currentWidth = 0
            currentHeight = 0
            currentRotation = 0
        }
    }
}
