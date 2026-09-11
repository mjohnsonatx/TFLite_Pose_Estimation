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

import android.graphics.PointF
import android.graphics.RectF
import org.tensorflow.lite.examples.poseestimation.data.KeyPoint
import org.tensorflow.lite.examples.poseestimation.data.Person

data class Track(
    val person: Person,
    val lastTimestamp: Long,
    val vx: Float = 0f,
    val vy: Float = 0f,
    val lastKeyPoints: List<KeyPoint> = person.keyPoints,
    val prevTimestamp: Long = lastTimestamp,
    val prevBoundingBox: RectF? = person.boundingBox
) {
    fun predictBoundingBox(timestamp: Long): RectF? {
        val box = person.boundingBox ?: return null
        val dt = timestamp - lastTimestamp
        if (dt <= 0 || (vx == 0f && vy == 0f)) return box
        val dx = (vx * dt).coerceIn(-1f, 1f)
        val dy = (vy * dt).coerceIn(-1f, 1f)
        return RectF(box.left + dx, box.top + dy, box.right + dx, box.bottom + dy)
    }

    fun predictKeyPoints(timestamp: Long): List<KeyPoint> {
        val dt = timestamp - lastTimestamp
        if (dt <= 0 || (vx == 0f && vy == 0f)) return person.keyPoints
        val dx = (vx * dt).coerceIn(-1f, 1f)
        val dy = (vy * dt).coerceIn(-1f, 1f)
        return person.keyPoints.map { kp ->
            KeyPoint(
                bodyPart = kp.bodyPart,
                coordinate = PointF(kp.coordinate.x + dx, kp.coordinate.y + dy),
                score = kp.score
            )
        }
    }
}
