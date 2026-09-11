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
 * Live state of one tracked identity.
 *
 * Besides the last observation a record keeps a short motion history so the tracker can predict
 * where the person should be *now*. That prediction is what lets two people swap sides without
 * swapping ids, and lets somebody who was occluded for a few frames pick their old id back up.
 *
 * The record is pure Kotlin and therefore unit testable; the Android facing `Track` wrapper is a
 * thin projection of it.
 */
class TrackRecord internal constructor(
    /** Stable, never reused identity of this track. */
    val id: Int,
    feature: PoseFeature,
    timestampMicros: Long,
    historySize: Int,
    maxExtrapolationMicros: Long
) {
    /** Most recent *observed* pose. Never a prediction. */
    var feature: PoseFeature = feature.copy()
        private set

    /** Timestamp in microseconds of the most recent observation. */
    var lastTimestamp: Long = timestampMicros
        private set

    /** Timestamp in microseconds of the first observation that created this track. */
    val createdTimestamp: Long = timestampMicros

    /** Number of frames this track was matched with a detection. */
    var hitCount: Int = 1
        private set

    /** Consecutive frames this track went unmatched. Reset on every hit. */
    var missCount: Int = 0
        private set

    private val stateSize = stateSizeOf(this.feature)
    private val motion = MotionEstimator(stateSize, historySize, maxExtrapolationMicros)
    private val stateScratch = FloatArray(stateSize)
    private val predictedState = FloatArray(stateSize)
    private val predictedKeyPoints = FloatArray(this.feature.keyPoints.size)
    private val predictedBox = if (this.feature.box != null) FloatArray(PoseFeature.BOX_VALUES) else null
    private val predictedFeature = PoseFeature(predictedKeyPoints, predictedBox, this.feature.score)

    init {
        writeState(this.feature, stateScratch)
        motion.record(stateScratch, timestampMicros)
    }

    /** Number of observations retained by the motion model. */
    val observationCount: Int get() = motion.observationCount

    /** Age of the last observation at [timestampMicros], in microseconds. */
    fun ageMicros(timestampMicros: Long): Long = timestampMicros - lastTimestamp

    /** Records a new observation and refreshes the motion model. */
    fun observe(feature: PoseFeature, timestampMicros: Long) {
        this.feature = feature.copy()
        lastTimestamp = timestampMicros
        hitCount++
        missCount = 0
        if (stateSizeOf(this.feature) == stateSize) {
            writeState(this.feature, stateScratch)
            motion.record(stateScratch, timestampMicros)
        } else {
            // The model changed shape mid-stream; the old motion history is meaningless.
            motion.reset()
        }
    }

    /** Marks this track as unmatched for the current frame. */
    fun miss() {
        missCount++
    }

    /**
     * Returns where this track is expected to be at [timestampMicros].
     *
     * @return a reusable [PoseFeature] owned by this record, or `null` when there is not enough
     * motion history. The result is only valid until the next call.
     */
    fun predict(timestampMicros: Long): PoseFeature? {
        if (feature.keyPoints.size != predictedKeyPoints.size) return null
        if (!motion.predict(predictedState, timestampMicros)) return null
        var offset = 0
        val box = predictedBox
        if (box != null) {
            predictedState.copyInto(box, 0, 0, PoseFeature.BOX_VALUES)
            offset = PoseFeature.BOX_VALUES
        }
        val observed = feature.keyPoints
        for (index in 0 until predictedFeature.keyPointCount) {
            val base = index * PoseFeature.VALUES_PER_KEY_POINT
            predictedKeyPoints[base] = predictedState[offset + index * 2]
            predictedKeyPoints[base + 1] = predictedState[offset + index * 2 + 1]
            // Confidence is a property of the detector, not something to extrapolate.
            predictedKeyPoints[base + 2] = observed[base + 2]
        }
        return predictedFeature
    }

    private companion object {
        /** Layout of the motion state vector: optional box, then `x`/`y` of every key point. */
        fun stateSizeOf(feature: PoseFeature): Int =
            (if (feature.box != null) PoseFeature.BOX_VALUES else 0) + feature.keyPointCount * 2

        fun writeState(feature: PoseFeature, out: FloatArray) {
            var offset = 0
            val box = feature.box
            if (box != null) {
                box.copyInto(out, 0, 0, PoseFeature.BOX_VALUES)
                offset = PoseFeature.BOX_VALUES
            }
            for (index in 0 until feature.keyPointCount) {
                out[offset + index * 2] = feature.x(index)
                out[offset + index * 2 + 1] = feature.y(index)
            }
        }
    }
}
