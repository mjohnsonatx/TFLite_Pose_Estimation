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

package org.tensorflow.lite.examples.poseestimation.imaging

import android.graphics.Bitmap
import android.graphics.ImageFormat
import android.media.Image
import androidx.core.graphics.createBitmap

/**
 * Thin Android adapter over [Yuv420Converter].
 *
 * All of the colour and geometry maths lives in framework independent classes; this type only
 * describes an [Image] as a [YuvFrame] and keeps the reusable pixel array and [Bitmap] alive
 * between frames. It replaces the RenderScript based converter, needs no `Context`, and works on
 * API 23 and above.
 *
 * Instances are not thread safe by design: a [ResourceGuard] rejects a second frame while the
 * previous one is still being analysed rather than letting inference read a half-written bitmap.
 */
class YuvToRgbConverter : AutoCloseable {

    private val guard = ResourceGuard("YuvToRgbConverter")
    private val pixelBuffers = FramePixelBufferPool()
    private var bitmap: Bitmap? = null

    /** Number of reusable bitmaps created so far. Reallocation only happens on geometry changes. */
    var bitmapAllocationCount: Int = 0
        private set

    /** Number of reusable pixel arrays created so far. */
    val pixelBufferAllocationCount: Int
        get() = pixelBuffers.allocationCount

    /**
     * Converts [image] into the reusable bitmap and runs [block] while that bitmap is guarded.
     *
     * The bitmap is owned by this converter and is overwritten by the next frame, so [block] must
     * not retain it. Holding the guard for the whole callback is what keeps an in-flight inference
     * from racing the next camera frame.
     *
     * @throws IllegalStateException if the converter is closed or already converting a frame.
     */
    fun <T> withConvertedFrame(image: Image, rotation: Rotation, block: (Bitmap) -> T): T =
        guard.withResource {
            val frame = image.toYuvFrame()
            val width = rotation.destinationWidth(frame.crop.width, frame.crop.height)
            val height = rotation.destinationHeight(frame.crop.width, frame.crop.height)
            val pixels = pixelBuffers.obtain(frame.crop.width, frame.crop.height).pixels
            Yuv420Converter.convert(frame, rotation, pixels)
            val target = obtainBitmap(width, height)
            target.setPixels(pixels, 0, width, 0, 0, width, height)
            block(target)
        }

    /**
     * Converts [image] into the caller supplied [output] bitmap.
     *
     * Kept for callers that manage their own bitmap. [output] must be mutable, `ARGB_8888` and
     * already sized to the rotated crop of [image].
     */
    @JvmOverloads
    fun yuvToRgb(image: Image, output: Bitmap, rotation: Rotation = Rotation.ROTATION_0) {
        guard.withResource {
            val frame = image.toYuvFrame()
            val width = rotation.destinationWidth(frame.crop.width, frame.crop.height)
            val height = rotation.destinationHeight(frame.crop.width, frame.crop.height)
            require(output.width == width && output.height == height) {
                "Output bitmap is ${output.width}x${output.height} but ${width}x$height is required"
            }
            val pixels = pixelBuffers.obtain(frame.crop.width, frame.crop.height).pixels
            Yuv420Converter.convert(frame, rotation, pixels)
            output.setPixels(pixels, 0, width, 0, 0, width, height)
        }
    }

    /**
     * Releases the reusable bitmap and pixel array.
     *
     * Waits briefly for an in-flight conversion so the native bitmap is never recycled underneath
     * a running inference pass. Safe to call more than once.
     */
    override fun close() {
        guard.awaitIdle(CLOSE_TIMEOUT_MILLIS)
        bitmap?.recycle()
        bitmap = null
        pixelBuffers.release()
    }

    private fun obtainBitmap(width: Int, height: Int): Bitmap {
        val existing = bitmap
        if (existing != null && !existing.isRecycled &&
            existing.width == width && existing.height == height
        ) {
            return existing
        }
        existing?.recycle()
        val created = createBitmap(width, height)
        bitmapAllocationCount++
        bitmap = created
        return created
    }

    private companion object {
        const val CLOSE_TIMEOUT_MILLIS = 250L
    }
}

/**
 * Describes an [Image] as a framework independent [YuvFrame] without copying any pixel data.
 *
 * @throws IllegalArgumentException when the image is not `YUV_420_888`.
 */
fun Image.toYuvFrame(): YuvFrame {
    require(format == ImageFormat.YUV_420_888) {
        "Expected YUV_420_888 but the image format was $format"
    }
    val imagePlanes = planes
    require(imagePlanes.size >= 3) { "YUV_420_888 requires three planes but got ${imagePlanes.size}" }
    val rect = cropRect
    // Some devices report an empty crop rectangle, which means "the whole image".
    val crop = if (rect == null || rect.isEmpty) {
        YuvCrop(0, 0, width, height)
    } else {
        YuvCrop(rect.left, rect.top, rect.right, rect.bottom)
    }
    return YuvFrame(
        width = width,
        height = height,
        yPlane = imagePlanes[0].let { YuvPlane(it.buffer, it.rowStride, it.pixelStride) },
        uPlane = imagePlanes[1].let { YuvPlane(it.buffer, it.rowStride, it.pixelStride) },
        vPlane = imagePlanes[2].let { YuvPlane(it.buffer, it.rowStride, it.pixelStride) },
        crop = crop
    )
}
