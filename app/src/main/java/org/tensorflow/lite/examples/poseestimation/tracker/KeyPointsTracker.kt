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

/**
 * KeypointTracker, which tracks poses based on keypoint similarity. This
 * tracker assumes that keypoints are provided in normalized image
 * coordinates.
 */
class KeyPointsTracker(
    trackerConfig: TrackerConfig = TrackerConfig(
        keyPointsTrackerParams = KeyPointsTrackerParams()
    )
) : AbstractTracker(trackerConfig) {

    override val similarityFunction: PoseSimilarityFunction =
        PoseSimilarity.keyPoints(trackerConfig.keyPointsTrackerParams)

    /**
     * Computes the Object Keypoint Similarity (OKS) between a person and track person.
     * This is similar in spirit to the calculation used by COCO keypoint eval:
     * https://cocodataset.org/#keypoints-eval
     * In this case, OKS is calculated as:
     * (1/sum_i d(c_i, c_ti)) * \sum_i exp(-d_i^2/(2*a_ti*x_i^2))*d(c_i, c_ti)
     * where
     *   d(x, y) is an indicator function which only produces 1 if x and y
     *     exceed a given threshold (i.e. keypointThreshold), otherwise 0.
     *   c_i is the confidence of keypoint i from the new person
     *   c_ti is the confidence of keypoint i from the track person
     *   d_i is the Euclidean distance between the person and track person keypoint
     *   a_ti is the area of the track object (the box covering the keypoints)
     *   x_i is a constant that controls falloff in a Gaussian distribution,
     *    computed as 2*keypointFalloff[i].
     * @param person1 A person.
     * @param person2 A track person.
     * @returns The OKS score between the person and the track person. This number is
     * between 0 and 1, and larger values indicate more keypoint similarity.
     */
    @VisibleForTesting(otherwise = VisibleForTesting.PRIVATE)
    fun oks(person1: Person, person2: Person): Float {
        val params = config.keyPointsTrackerParams ?: return 0f
        return PoseSimilarity.oks(person1.toPoseFeature(), person2.toPoseFeature(), params)
    }

    /**
     * Computes the area of a bounding box that tightly covers keypoints.
     * @param keyPoints A list of KeyPoint.
     * @returns The area of the object.
     */
    @VisibleForTesting(otherwise = VisibleForTesting.PRIVATE)
    fun area(keyPoints: List<KeyPoint>): Float {
        val threshold = config.keyPointsTrackerParams?.keypointThreshold ?: 0f
        val person = Person(keyPoints = keyPoints, score = 0f)
        return PoseSimilarity.keyPointArea(person.toPoseFeature(), threshold)
    }
}
