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
 * A reusable, fixed size ARGB scratch buffer for one frame.
 *
 * Camera previews deliver a constant resolution, so the conversion target can be allocated once
 * and rewritten in place instead of producing a multi-megabyte array per frame.
 */
class FramePixelBuffer(val width: Int, val height: Int) {

    init {
        require(width > 0 && height > 0) {
            "Frame buffer must not be empty but was ${width}x$height"
        }
    }

    /** Packed ARGB_8888 pixels, row-major with a stride of [width]. */
    val pixels: IntArray = IntArray(width * height)

    /** Number of pixels held by this buffer. */
    val size: Int get() = pixels.size

    /** True when this buffer can hold a [width] x [height] frame exactly. */
    fun matches(width: Int, height: Int): Boolean = this.width == width && this.height == height

    /** Clears the buffer so a released frame cannot be observed by the next consumer. */
    fun zero() {
        pixels.fill(0)
    }
}

/**
 * Owns the single [FramePixelBuffer] used by the conversion pipeline.
 *
 * A new array is only allocated when the requested geometry changes, which happens when the camera
 * or the preview size changes. Steady-state frame processing reuses the same array forever, and
 * [release] drops it so the memory is reclaimed during teardown.
 */
class FramePixelBufferPool {

    private var buffer: FramePixelBuffer? = null

    /** Number of arrays this pool has allocated. Used by tests to prove frames do not allocate. */
    var allocationCount: Int = 0
        private set

    /** The currently held buffer, or `null` when the pool is empty. */
    fun peek(): FramePixelBuffer? = buffer

    /**
     * Returns a buffer of exactly [width] x [height], reusing the existing one when possible.
     *
     * The previous buffer is dropped when the geometry changes so a stale, differently sized array
     * cannot leak into the frame path.
     */
    fun obtain(width: Int, height: Int): FramePixelBuffer {
        val existing = buffer
        if (existing != null && existing.matches(width, height)) {
            return existing
        }
        val created = FramePixelBuffer(width, height)
        allocationCount++
        buffer = created
        return created
    }

    /** Zeroes and drops the pooled buffer. Safe to call repeatedly, including from `onDestroy`. */
    fun release() {
        buffer?.zero()
        buffer = null
    }
}
