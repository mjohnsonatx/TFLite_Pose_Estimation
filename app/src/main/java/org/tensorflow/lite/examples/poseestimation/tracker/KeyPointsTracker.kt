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

import androidx.annotation.VisibleForTesting
import org.tensorflow.lite.examples.poseestimation.data.KeyPoint
import org.tensorflow.lite.examples.poseestimation.data.Person
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow

/**
 * KeypointTracker, which tracks poses based on keypoint similarity (OKS)
 * with motion prediction.
 */
class KeyPointsTracker(
    trackerConfig: TrackerConfig = TrackerConfig(
        keyPointsTrackerParams = KeyPointsTrackerParams()
    )
) : AbstractTracker(trackerConfig) {

    /**
     * Computes similarity based on Object Keypoint Similarity (OKS).
     */
    override fun computeSimilarity(persons: List<Person>): List<List<Float>> {
        if (persons.isEmpty() || tracks.isEmpty()) {
            return emptyList()
        }
        return persons.map { person -> tracks.map { track -> oks(person, track.person) } }
    }

    /**
     * Computes similarity using motion-predicted keypoint positions at the given timestamp.
     */
    override fun computeSimilarity(persons: List<Person>, timestamp: Long): List<List<Float>> {
        if (persons.isEmpty() || tracks.isEmpty()) {
            return emptyList()
        }
        return persons.map { person ->
            tracks.map { track ->
                val predictedKeypoints = track.predictKeyPoints(timestamp)
                val predOks = oks(person.keyPoints, predictedKeypoints)
                val lastOks = oks(person.keyPoints, track.person.keyPoints)
                max(predOks, lastOks)
            }
        }
    }

    /**
     * Computes the Object Keypoint Similarity (OKS) between two persons.
     */
    @VisibleForTesting(otherwise = VisibleForTesting.PRIVATE)
    fun oks(person1: Person, person2: Person): Float {
        return oks(person1.keyPoints, person2.keyPoints)
    }

    /**
     * Computes the Object Keypoint Similarity (OKS) between two sets of keypoints.
     */
    @VisibleForTesting(otherwise = VisibleForTesting.PRIVATE)
    fun oks(keyPoints1: List<KeyPoint>, keyPoints2: List<KeyPoint>): Float {
        if (config.keyPointsTrackerParams == null) return 0f
        val boxArea = area(keyPoints2) + 1e-6
        var oksTotal = 0f
        var numValidKeyPoints = 0

        keyPoints1.forEachIndexed { index, poseKpt ->
            if (index >= keyPoints2.size) return@forEachIndexed
            val trackKpt = keyPoints2[index]
            val threshold = config.keyPointsTrackerParams.keypointThreshold
            if (poseKpt.score < threshold || trackKpt.score < threshold) {
                return@forEachIndexed
            }
            numValidKeyPoints += 1
            val dSquared: Float =
                (poseKpt.coordinate.x - trackKpt.coordinate.x).pow(2) +
                (poseKpt.coordinate.y - trackKpt.coordinate.y).pow(2)
            val x = 2f * config.keyPointsTrackerParams.keypointFalloff[index]
            oksTotal += exp(-1f * dSquared / (2f * boxArea * x.pow(2))).toFloat()
        }
        if (numValidKeyPoints < config.keyPointsTrackerParams.minNumKeyPoints) {
            return 0f
        }
        return oksTotal / numValidKeyPoints
    }

    /**
     * Computes the area of a bounding box that tightly covers keypoints.
     */
    @VisibleForTesting(otherwise = VisibleForTesting.PRIVATE)
    fun area(keyPoints: List<KeyPoint>): Float {
        val validKeypoint = keyPoints.filter {
            it.score > (config.keyPointsTrackerParams?.keypointThreshold ?: 0f)
        }
        if (validKeypoint.isEmpty()) return 0f
        val minX = min(1f, validKeypoint.minOf { it.coordinate.x })
        val maxX = max(0f, validKeypoint.maxOf { it.coordinate.x })
        val minY = min(1f, validKeypoint.minOf { it.coordinate.y })
        val maxY = max(0f, validKeypoint.maxOf { it.coordinate.y })
        return (maxX - minX) * (maxY - minY)
    }
}
