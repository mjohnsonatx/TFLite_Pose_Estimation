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

import android.graphics.RectF
import androidx.annotation.VisibleForTesting
import org.tensorflow.lite.examples.poseestimation.data.Person
import kotlin.math.max
import kotlin.math.min

/**
 * BoundingBoxTracker, which tracks objects based on bounding box similarity,
 * defined as intersection-over-union (IoU) with motion prediction.
 */
class BoundingBoxTracker(config: TrackerConfig = TrackerConfig()) : AbstractTracker(config) {

    /**
     * Computes similarity based on intersection-over-union (IoU).
     */
    override fun computeSimilarity(persons: List<Person>): List<List<Float>> {
        if (persons.isEmpty() || tracks.isEmpty()) {
            return emptyList()
        }
        return persons.map { person -> tracks.map { track -> iou(person, track.person) } }
    }

    /**
     * Computes similarity using motion-predicted positions at the given timestamp.
     */
    override fun computeSimilarity(persons: List<Person>, timestamp: Long): List<List<Float>> {
        if (persons.isEmpty() || tracks.isEmpty()) {
            return emptyList()
        }
        return persons.map { person ->
            tracks.map { track ->
                val predictedBox = track.predictBoundingBox(timestamp)
                val predIou = iou(person.boundingBox, predictedBox)
                val lastIou = iou(person.boundingBox, track.person.boundingBox)
                max(predIou, lastIou)
            }
        }
    }

    /**
     * Computes the intersection-over-union (IoU) between two person instances.
     */
    @VisibleForTesting(otherwise = VisibleForTesting.PRIVATE)
    fun iou(person1: Person, person2: Person): Float {
        return iou(person1.boundingBox, person2.boundingBox)
    }

    /**
     * Computes the intersection-over-union (IoU) between two bounding boxes.
     */
    @VisibleForTesting(otherwise = VisibleForTesting.PRIVATE)
    fun iou(box1: RectF?, box2: RectF?): Float {
        if (box1 != null && box2 != null) {
            val xMin = max(box1.left, box2.left)
            val yMin = max(box1.top, box2.top)
            val xMax = min(box1.right, box2.right)
            val yMax = min(box1.bottom, box2.bottom)
            if (xMin >= xMax || yMin >= yMax) return 0f
            val intersection = (xMax - xMin) * (yMax - yMin)
            val area1 = box1.width() * box1.height()
            val area2 = box2.width() * box2.height()
            val union = area1 + area2 - intersection
            if (union <= 0f) return 0f
            return intersection / union
        }
        return 0f
    }
}
