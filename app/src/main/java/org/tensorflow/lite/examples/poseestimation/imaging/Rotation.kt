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
 * A right-angle, clockwise rotation applied while a frame is written into its destination buffer.
 *
 * This type is deliberately free of Android dependencies so that the rotation mapping can be
 * verified by local JVM unit tests.
 */
enum class Rotation(val degrees: Int) {
    ROTATION_0(0),
    ROTATION_90(90),
    ROTATION_180(180),
    ROTATION_270(270);

    /** Width of the destination image produced from a [sourceWidth] x [sourceHeight] source. */
    fun destinationWidth(sourceWidth: Int, sourceHeight: Int): Int =
        if (swapsAxes) sourceHeight else sourceWidth

    /** Height of the destination image produced from a [sourceWidth] x [sourceHeight] source. */
    fun destinationHeight(sourceWidth: Int, sourceHeight: Int): Int =
        if (swapsAxes) sourceWidth else sourceHeight

    /** True when the rotation exchanges the width and height of the image. */
    val swapsAxes: Boolean
        get() = this == ROTATION_90 || this == ROTATION_270

    /**
     * Maps the source pixel at ([x], [y]) to a linear index inside the destination buffer.
     *
     * The destination buffer is laid out row-major with a stride of
     * [destinationWidth]([sourceWidth], [sourceHeight]).
     */
    fun destinationIndex(x: Int, y: Int, sourceWidth: Int, sourceHeight: Int): Int = when (this) {
        ROTATION_0 -> y * sourceWidth + x
        // Clockwise: (x, y) -> (sourceHeight - 1 - y, x); destination stride is sourceHeight.
        ROTATION_90 -> x * sourceHeight + (sourceHeight - 1 - y)
        ROTATION_180 -> (sourceHeight - 1 - y) * sourceWidth + (sourceWidth - 1 - x)
        // Counter-clockwise: (x, y) -> (y, sourceWidth - 1 - x); destination stride is sourceHeight.
        ROTATION_270 -> (sourceWidth - 1 - x) * sourceHeight + y
    }

    companion object {
        /**
         * Returns the [Rotation] for [degrees], which must be a multiple of 90. Negative values and
         * values greater than 360 are normalised.
         */
        @JvmStatic
        fun ofDegrees(degrees: Int): Rotation {
            require(degrees % 90 == 0) { "Rotation must be a multiple of 90 degrees but was $degrees" }
            return when (((degrees % 360) + 360) % 360) {
                0 -> ROTATION_0
                90 -> ROTATION_90
                180 -> ROTATION_180
                else -> ROTATION_270
            }
        }
    }
}
