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

/**
 * Converts `YUV_420_888` frames to packed ARGB_8888 pixels.
 *
 * This replaces the RenderScript `ScriptIntrinsicYuvToRGB` pipeline that used to back the camera
 * preview. RenderScript was deprecated in API 31 and removed from the NDK, and the old
 * implementation additionally required `ByteBuffer.array()`, which is unavailable for the direct
 * buffers that `android.media.Image` hands out.
 *
 * The implementation is pure Kotlin with no Android or third party dependencies, works on API 23
 * and above, and is therefore covered by local JVM unit tests.
 *
 * Colours use the limited range BT.601 integer approximation:
 * ```
 * C = max(0, Y - 16)   D = U - 128   E = V - 128
 * R = clamp((298*C + 409*E + 128) >> 8)
 * G = clamp((298*C - 100*D - 208*E + 128) >> 8)
 * B = clamp((298*C + 516*D + 128) >> 8)
 * ```
 * Every chroma sample covers the 2x2 luma block of the *uncropped* source image, so crop
 * rectangles with an odd origin keep sampling the same chroma grid the camera produced.
 */
object Yuv420Converter {

    /** Fully opaque alpha channel, pre-shifted into position. */
    private const val OPAQUE_ALPHA = 0xFF shl 24

    /**
     * Converts a single YUV triple to an opaque ARGB pixel.
     *
     * @param y luma sample as an unsigned byte in `[0, 255]`.
     * @param u blue-difference chroma sample as an unsigned byte in `[0, 255]`.
     * @param v red-difference chroma sample as an unsigned byte in `[0, 255]`.
     */
    fun yuvToArgb(y: Int, u: Int, v: Int): Int {
        val c = if (y < 16) 0 else y - 16
        val d = u - 128
        val e = v - 128
        val r = clampToByte((298 * c + 409 * e + 128) shr 8)
        val g = clampToByte((298 * c - 100 * d - 208 * e + 128) shr 8)
        val b = clampToByte((298 * c + 516 * d + 128) shr 8)
        return OPAQUE_ALPHA or (r shl 16) or (g shl 8) or b
    }

    /** Number of ARGB pixels [convert] writes for [frame]. Rotation never changes the count. */
    fun requiredCapacity(frame: YuvFrame): Int = frame.pixelCount

    /**
     * Converts the cropped region of [frame] into [destination] applying [rotation].
     *
     * [destination] must hold at least [requiredCapacity] pixels; supplying a longer, reused array
     * is expected and leaves the trailing pixels untouched. Nothing is allocated, which keeps the
     * per-frame path free of garbage.
     *
     * Output pixel `(x, y)` of the unrotated image corresponds to source pixel
     * `(crop.left + x, crop.top + y)`; [rotation] only changes where that pixel is written.
     *
     * @throws IllegalArgumentException if [destination] is too small.
     */
    fun convert(frame: YuvFrame, rotation: Rotation, destination: IntArray) {
        val crop = frame.crop
        val cropWidth = crop.width
        val cropHeight = crop.height
        val required = cropWidth * cropHeight
        require(destination.size >= required) {
            "Destination holds ${destination.size} pixels but ${cropWidth}x$cropHeight " +
                "($required pixels) are required"
        }

        val yPlane = frame.yPlane
        val uPlane = frame.uPlane
        val vPlane = frame.vPlane

        // Walking the destination with constant strides keeps the rotation branch out of the
        // inner loop while still matching Rotation.destinationIndex exactly.
        val strideX = destinationStrideX(rotation, cropWidth, cropHeight)
        val strideY = destinationStrideY(rotation, cropWidth, cropHeight)
        var rowStart = rotation.destinationIndex(0, 0, cropWidth, cropHeight)

        for (y in 0 until cropHeight) {
            val sourceY = crop.top + y
            val chromaRow = sourceY shr 1
            var destinationIndex = rowStart
            var chromaColumn = -1
            var u = 0
            var v = 0
            for (x in 0 until cropWidth) {
                val sourceX = crop.left + x
                val column = sourceX shr 1
                // Each chroma sample is shared by two horizontally adjacent luma samples.
                if (column != chromaColumn) {
                    chromaColumn = column
                    u = uPlane.sample(column, chromaRow)
                    v = vPlane.sample(column, chromaRow)
                }
                destination[destinationIndex] =
                    yuvToArgb(yPlane.sample(sourceX, sourceY), u, v)
                destinationIndex += strideX
            }
            rowStart += strideY
        }
    }

    /** Destination index delta produced by advancing one source pixel to the right. */
    internal fun destinationStrideX(rotation: Rotation, cropWidth: Int, cropHeight: Int): Int =
        when (rotation) {
            Rotation.ROTATION_0 -> 1
            Rotation.ROTATION_90 -> cropHeight
            Rotation.ROTATION_180 -> -1
            Rotation.ROTATION_270 -> -cropHeight
        }

    /** Destination index delta produced by advancing one source row downwards. */
    internal fun destinationStrideY(rotation: Rotation, cropWidth: Int, cropHeight: Int): Int =
        when (rotation) {
            Rotation.ROTATION_0 -> cropWidth
            Rotation.ROTATION_90 -> -1
            Rotation.ROTATION_180 -> -cropWidth
            Rotation.ROTATION_270 -> 1
        }

    private fun clampToByte(value: Int): Int = when {
        value < 0 -> 0
        value > 255 -> 255
        else -> value
    }
}
