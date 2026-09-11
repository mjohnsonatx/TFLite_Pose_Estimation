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

import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

/** Scores how likely it is that a detection and a track describe the same person. */
fun interface PoseSimilarityFunction {
    /** Returns a score in `[0, 1]`; larger values mean more similar. */
    fun similarity(detection: PoseFeature, track: PoseFeature): Float
}

/**
 * Pure Kotlin similarity measures shared by the trackers.
 *
 * The formulas are byte-for-byte the ones the original `BoundingBoxTracker` and `KeyPointsTracker`
 * used, extracted so they can be unit tested without the Android framework.
 */
object PoseSimilarity {

    /** Intersection over union of two bounding boxes; `0` when either pose has no box. */
    val IOU = PoseSimilarityFunction { detection, track -> iou(detection, track) }

    /**
     * Intersection-over-union between two poses.
     *
     * @return a value in `[0, 1]`, or `0` when either pose has no bounding box or they do not
     * overlap.
     */
    fun iou(detection: PoseFeature, track: PoseFeature): Float {
        if (detection.box == null || track.box == null) return 0f
        val xMin = max(detection.left, track.left)
        val yMin = max(detection.top, track.top)
        val xMax = min(detection.right, track.right)
        val yMax = min(detection.bottom, track.bottom)
        if (xMin >= xMax || yMin >= yMax) return 0f
        val intersection = (xMax - xMin) * (yMax - yMin)
        val detectionArea = (detection.right - detection.left) * (detection.bottom - detection.top)
        val trackArea = (track.right - track.left) * (track.bottom - track.top)
        val union = detectionArea + trackArea - intersection
        if (union <= 0f) return 0f
        return intersection / union
    }

    /**
     * Object Keypoint Similarity, following the COCO key point evaluation protocol.
     *
     * Key points are expected in normalised image coordinates. Only key points that clear
     * [KeyPointsTrackerParams.keypointThreshold] in *both* poses contribute; when fewer than
     * [KeyPointsTrackerParams.minNumKeyPoints] survive, the poses are considered unrelated.
     */
    fun oks(
        detection: PoseFeature,
        track: PoseFeature,
        params: KeyPointsTrackerParams
    ): Float {
        val count = min(detection.keyPointCount, track.keyPointCount)
        if (count == 0) return 0f
        val boxArea = keyPointArea(track, params.keypointThreshold) + 1e-6
        var oksTotal = 0f
        var validKeyPoints = 0
        for (index in 0 until count) {
            if (index >= params.keypointFalloff.size) break
            if (detection.keyPointScore(index) < params.keypointThreshold ||
                track.keyPointScore(index) < params.keypointThreshold
            ) {
                continue
            }
            validKeyPoints++
            val dx = detection.x(index) - track.x(index)
            val dy = detection.y(index) - track.y(index)
            val squaredDistance = dx * dx + dy * dy
            val falloff = 2f * params.keypointFalloff[index]
            oksTotal += exp(-1f * squaredDistance / (2f * boxArea * falloff * falloff)).toFloat()
        }
        if (validKeyPoints < params.minNumKeyPoints) return 0f
        return oksTotal / validKeyPoints
    }

    /**
     * Area of the tightest box that covers every key point scoring above [threshold].
     *
     * The coordinates are clamped to the unit square the way the original implementation did, so
     * poses that leave the frame do not inflate the OKS falloff.
     */
    fun keyPointArea(pose: PoseFeature, threshold: Float): Float {
        var minX = Float.MAX_VALUE
        var maxX = -Float.MAX_VALUE
        var minY = Float.MAX_VALUE
        var maxY = -Float.MAX_VALUE
        var valid = 0
        for (index in 0 until pose.keyPointCount) {
            if (pose.keyPointScore(index) <= threshold) continue
            valid++
            val x = pose.x(index)
            val y = pose.y(index)
            if (x < minX) minX = x
            if (x > maxX) maxX = x
            if (y < minY) minY = y
            if (y > maxY) maxY = y
        }
        if (valid == 0) return 0f
        return (max(0f, maxX) - min(1f, minX)) * (max(0f, maxY) - min(1f, minY))
    }

    /** Builds the OKS similarity function for [params]. */
    fun keyPoints(params: KeyPointsTrackerParams?): PoseSimilarityFunction =
        if (params == null) {
            PoseSimilarityFunction { _, _ -> 0f }
        } else {
            PoseSimilarityFunction { detection, track -> oks(detection, track, params) }
        }
}
