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

package org.tensorflow.lite.examples.poseestimation.tracker

/**
 * Framework independent view of one detected pose.
 *
 * `Person` carries `android.graphics` types, which cannot be instantiated in a local JVM unit test.
 * The tracking engine therefore works on this flat representation so the whole matching, motion and
 * ageing pipeline can be exercised off-device. Coordinates are whatever the caller provides; the
 * MoveNet MultiPose path keeps using normalised `[0, 1]` image coordinates.
 *
 * @param keyPoints three floats per key point: `x`, `y`, `score`.
 * @param box optional bounding box as `left`, `top`, `right`, `bottom`, or `null` when the model
 * does not produce one.
 * @param score detection confidence.
 */
class PoseFeature(
    val keyPoints: FloatArray,
    val box: FloatArray? = null,
    val score: Float = 0f
) {
    init {
        require(keyPoints.size % VALUES_PER_KEY_POINT == 0) {
            "keyPoints must hold $VALUES_PER_KEY_POINT values per point but had ${keyPoints.size}"
        }
        require(box == null || box.size == BOX_VALUES) {
            "box must hold $BOX_VALUES values but had ${box?.size}"
        }
    }

    /** Number of key points described by this pose. */
    val keyPointCount: Int get() = keyPoints.size / VALUES_PER_KEY_POINT

    fun x(index: Int): Float = keyPoints[index * VALUES_PER_KEY_POINT]

    fun y(index: Int): Float = keyPoints[index * VALUES_PER_KEY_POINT + 1]

    fun keyPointScore(index: Int): Float = keyPoints[index * VALUES_PER_KEY_POINT + 2]

    val left: Float get() = box?.get(0) ?: 0f
    val top: Float get() = box?.get(1) ?: 0f
    val right: Float get() = box?.get(2) ?: 0f
    val bottom: Float get() = box?.get(3) ?: 0f

    /** Deep copy, so a stored track state can never alias a caller owned array. */
    fun copy(): PoseFeature = PoseFeature(keyPoints.copyOf(), box?.copyOf(), score)

    companion object {
        const val VALUES_PER_KEY_POINT = 3
        const val BOX_VALUES = 4

        /** Convenience factory for a pose that only carries a bounding box. */
        @JvmStatic
        fun ofBox(left: Float, top: Float, right: Float, bottom: Float, score: Float = 1f) =
            PoseFeature(FloatArray(0), floatArrayOf(left, top, right, bottom), score)
    }
}
