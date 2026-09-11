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

import java.nio.ByteBuffer

/**
 * A read-only description of a single `YUV_420_888` image plane.
 *
 * The plane is addressed with absolute [ByteBuffer] reads relative to [baseOffset], which is the
 * buffer position observed when this plane was created. Nothing is ever copied out of the buffer
 * and no backing array is required, so direct buffers handed out by `android.media.Image` work
 * unchanged.
 */
class YuvPlane(
    private val buffer: ByteBuffer,
    val rowStride: Int,
    val pixelStride: Int,
    val baseOffset: Int = buffer.position()
) {
    init {
        require(rowStride > 0) { "rowStride must be positive but was $rowStride" }
        require(pixelStride > 0) { "pixelStride must be positive but was $pixelStride" }
        require(baseOffset >= 0) { "baseOffset must not be negative but was $baseOffset" }
        require(baseOffset <= buffer.limit()) {
            "baseOffset $baseOffset is outside of the buffer limit ${buffer.limit()}"
        }
    }

    /** Number of readable bytes from [baseOffset] to the buffer limit. */
    val capacity: Int
        get() = buffer.limit() - baseOffset

    /**
     * Returns the unsigned sample stored at [column], [row] of this plane.
     *
     * Reading is absolute so the buffer position, mark and limit are never mutated. That keeps the
     * converter safe to call from several threads with the same plane and avoids the hidden
     * allocation of `ByteBuffer.array()`.
     */
    fun sample(column: Int, row: Int): Int {
        val index = baseOffset + row * rowStride + column * pixelStride
        return buffer.get(index).toInt() and 0xFF
    }

    /**
     * Verifies that every sample of a [columns] x [rows] plane can be read.
     *
     * The last sample of the last row bounds the whole plane, so only that offset is checked. Some
     * devices hand out a Y plane whose final row is not padded to `rowStride`, which is why the
     * trailing padding of the last row is deliberately not required to be present.
     */
    fun requireCapacityFor(columns: Int, rows: Int, name: String) {
        require(columns > 0 && rows > 0) {
            "Plane $name must contain at least one sample but was ${columns}x$rows"
        }
        val lastSample = (rows - 1) * rowStride + (columns - 1) * pixelStride
        require(lastSample < capacity) {
            "Plane $name needs ${lastSample + 1} bytes from offset $baseOffset but only " +
                "$capacity are available (rowStride=$rowStride, pixelStride=$pixelStride, " +
                "size=${columns}x$rows)"
        }
    }
}

/**
 * An inclusive-exclusive crop rectangle expressed in source pixels.
 *
 * Mirrors `android.graphics.Rect` semantics without depending on the Android framework.
 */
data class YuvCrop(val left: Int, val top: Int, val right: Int, val bottom: Int) {
    val width: Int get() = right - left
    val height: Int get() = bottom - top

    init {
        require(left >= 0 && top >= 0) { "Crop origin must not be negative but was ($left, $top)" }
        require(width > 0 && height > 0) { "Crop must not be empty but was ${width}x$height" }
    }
}

/**
 * An immutable, framework independent view over a `YUV_420_888` frame.
 *
 * [width] and [height] describe the *uncropped* source image; chroma sub-sampling is always
 * resolved against those coordinates, which is what makes odd crop origins behave correctly.
 */
class YuvFrame(
    val width: Int,
    val height: Int,
    val yPlane: YuvPlane,
    val uPlane: YuvPlane,
    val vPlane: YuvPlane,
    val crop: YuvCrop = YuvCrop(0, 0, width, height)
) {
    init {
        require(width > 0 && height > 0) { "Frame must not be empty but was ${width}x$height" }
        require(crop.right <= width && crop.bottom <= height) {
            "Crop $crop does not fit inside a ${width}x$height frame"
        }
        yPlane.requireCapacityFor(width, height, "Y")
        val chromaColumns = (width + 1) / 2
        val chromaRows = (height + 1) / 2
        uPlane.requireCapacityFor(chromaColumns, chromaRows, "U")
        vPlane.requireCapacityFor(chromaColumns, chromaRows, "V")
    }

    /** Number of ARGB pixels produced when this frame is converted. */
    val pixelCount: Int
        get() = crop.width * crop.height
}
