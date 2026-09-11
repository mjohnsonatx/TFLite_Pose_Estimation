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
 * Tuning for [PoseTrackerEngine] and the trackers built on top of it.
 *
 * The first four parameters keep their original position and meaning so existing call sites and
 * tests continue to compile.
 *
 * @param maxTracks how many identities may be alive at once.
 * @param maxAge milliseconds a track survives without being observed. A person who reappears within
 * this window can resume their identity; once the window passes the track is deleted for good.
 * @param minSimilarity scores below this are treated as "not the same person" and never produce a
 * match, even when that leaves a detection and a track both unmatched.
 * @param keyPointsTrackerParams OKS parameters, only used by [KeyPointsTracker].
 * @param maxDetections upper bound on the detections considered per frame. MoveNet MultiPose
 * reports at most six people, and bounding the matrix bounds the matching cost.
 * @param motionHistorySize observations retained per track for velocity estimation.
 * @param maxExtrapolationMillis how far a track may be extrapolated past its last observation.
 * Prediction is clamped so a long occlusion does not fling the predicted pose off screen.
 */
data class TrackerConfig(
    val maxTracks: Int = MAX_TRACKS,
    val maxAge: Int = MAX_AGE,
    val minSimilarity: Float = MIN_SIMILARITY,
    val keyPointsTrackerParams: KeyPointsTrackerParams? = null,
    val maxDetections: Int = MAX_DETECTIONS,
    val motionHistorySize: Int = MOTION_HISTORY_SIZE,
    val maxExtrapolationMillis: Int = MAX_EXTRAPOLATION_MILLIS
) {
    init {
        require(maxTracks > 0) { "maxTracks must be positive but was $maxTracks" }
        require(maxAge > 0) { "maxAge must be positive but was $maxAge" }
        require(maxDetections > 0) { "maxDetections must be positive but was $maxDetections" }
        require(motionHistorySize >= 2) {
            "motionHistorySize must be at least 2 but was $motionHistorySize"
        }
        require(maxExtrapolationMillis >= 0) {
            "maxExtrapolationMillis must not be negative but was $maxExtrapolationMillis"
        }
    }

    companion object {
        /** Live identities supported at once. */
        const val MAX_TRACKS = 18

        /** Milliseconds. */
        const val MAX_AGE = 1000

        const val MIN_SIMILARITY = 0.15f

        /** Detections MoveNet MultiPose can report in one frame. */
        const val MAX_DETECTIONS = 6

        const val MOTION_HISTORY_SIZE = 5

        /** Milliseconds. */
        const val MAX_EXTRAPOLATION_MILLIS = 300
    }
}

data class KeyPointsTrackerParams(
    val keypointThreshold: Float = KEYPOINT_THRESHOLD,
    // List of per-keypoint standard deviation `σ`, keypoints on a person's body (shoulders, knees, hips, etc.)
    // tend to have a `σ` much larger than on a person's head (eyes, nose, ears).
    // Read more at: https://cocodataset.org/#keypoints-eval
    val keypointFalloff: List<Float> = KEYPOINT_FALLOFF,
    val minNumKeyPoints: Int = MIN_NUM_KEYPOINT
) {
    companion object {
        // From COCO:
        // https://cocodataset.org/#keypoints-eval
        private val KEYPOINT_FALLOFF: List<Float> = listOf(
            0.026f, 0.025f, 0.025f, 0.035f, 0.035f, 0.079f, 0.079f, 0.072f, 0.072f, 0.062f,
            0.062f, 0.107f, 0.107f, 0.087f, 0.087f, 0.089f, 0.089f
        )
        private const val KEYPOINT_THRESHOLD = 0.3f
        private const val MIN_NUM_KEYPOINT = 4
    }
}
