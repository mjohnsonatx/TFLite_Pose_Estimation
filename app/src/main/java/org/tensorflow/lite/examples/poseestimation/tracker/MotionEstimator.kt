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
 * Constant-velocity motion model over a short ring buffer of timestamped observations.
 *
 * Velocity is expressed per **microsecond** and derived from the oldest and newest samples still in
 * the window, so it is completely independent of the frame rate. A pipeline that drops frames under
 * load, or a camera that delivers jittery intervals, still produces the same physical velocity.
 *
 * Extrapolation is clamped to [maxExtrapolationMicros]. Without that clamp a track that has been
 * unobserved for most of `maxAge` would be predicted far outside the frame and would never be able
 * to re-acquire its identity when the person reappears.
 *
 * The estimator allocates its storage once and never allocates while recording or predicting.
 */
class MotionEstimator(
    private val dimension: Int,
    private val historySize: Int = DEFAULT_HISTORY_SIZE,
    private val maxExtrapolationMicros: Long = DEFAULT_MAX_EXTRAPOLATION_MICROS
) {
    init {
        require(dimension >= 0) { "dimension must not be negative but was $dimension" }
        require(historySize >= 2) { "historySize must be at least 2 but was $historySize" }
        require(maxExtrapolationMicros >= 0) { "maxExtrapolationMicros must not be negative" }
    }

    private val samples = Array(historySize) { FloatArray(dimension) }
    private val timestamps = LongArray(historySize)
    private val velocityScratch = FloatArray(dimension)

    private var count = 0
    private var newest = -1

    /** Number of observations currently held, capped at `historySize`. */
    val observationCount: Int get() = count

    /** Timestamp of the most recent observation in microseconds, or [Long.MIN_VALUE] if empty. */
    val lastTimestampMicros: Long
        get() = if (count == 0) Long.MIN_VALUE else timestamps[newest]

    /** True when at least two observations with distinct timestamps are available. */
    val hasVelocity: Boolean
        get() = count >= 2 && timestamps[newest] > timestamps[oldestIndex()]

    /**
     * Records an observation.
     *
     * Out-of-order samples (a timestamp at or before the newest one) overwrite the newest slot
     * instead of corrupting the velocity estimate with a negative time delta.
     */
    fun record(values: FloatArray, timestampMicros: Long) {
        require(values.size == dimension) {
            "Expected $dimension values but got ${values.size}"
        }
        if (count > 0 && timestampMicros <= timestamps[newest]) {
            values.copyInto(samples[newest])
            timestamps[newest] = timestampMicros
            return
        }
        newest = if (count == 0) 0 else (newest + 1) % historySize
        values.copyInto(samples[newest])
        timestamps[newest] = timestampMicros
        if (count < historySize) count++
    }

    /** Copies the newest observation into [out]. Returns `false` when nothing has been recorded. */
    fun latest(out: FloatArray): Boolean {
        require(out.size == dimension) { "Expected $dimension values but got ${out.size}" }
        if (count == 0) return false
        samples[newest].copyInto(out)
        return true
    }

    /**
     * Copies the per-microsecond velocity into [out].
     *
     * @return `false` when there is not enough history, leaving [out] untouched.
     */
    fun velocity(out: FloatArray): Boolean {
        require(out.size == dimension) { "Expected $dimension values but got ${out.size}" }
        if (!hasVelocity) return false
        val oldest = oldestIndex()
        val span = (timestamps[newest] - timestamps[oldest]).toDouble()
        val from = samples[oldest]
        val to = samples[newest]
        for (index in 0 until dimension) {
            out[index] = ((to[index] - from[index]) / span).toFloat()
        }
        return true
    }

    /**
     * Writes the state predicted for [timestampMicros] into [out].
     *
     * @return `false` when no velocity is available; the caller should fall back to the last
     * observation in that case.
     */
    fun predict(out: FloatArray, timestampMicros: Long): Boolean {
        require(out.size == dimension) { "Expected $dimension values but got ${out.size}" }
        if (!velocity(velocityScratch)) return false
        val elapsed = (timestampMicros - timestamps[newest])
            .coerceIn(0L, maxExtrapolationMicros)
            .toFloat()
        val latest = samples[newest]
        for (index in 0 until dimension) {
            out[index] = latest[index] + velocityScratch[index] * elapsed
        }
        return true
    }

    /** Drops all history. */
    fun reset() {
        count = 0
        newest = -1
    }

    private fun oldestIndex(): Int =
        if (count < historySize) 0 else (newest + 1) % historySize

    companion object {
        const val DEFAULT_HISTORY_SIZE = 5

        /** 300 ms. Long enough to bridge a short occlusion, short enough to stay believable. */
        const val DEFAULT_MAX_EXTRAPOLATION_MICROS = 300_000L
    }
}
