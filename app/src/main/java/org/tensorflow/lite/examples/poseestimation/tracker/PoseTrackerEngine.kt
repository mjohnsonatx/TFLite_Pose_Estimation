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

import kotlin.math.max
import kotlin.math.min

/**
 * Multi-object pose tracker with stable identities.
 *
 * Every frame goes through the same five steps:
 * 1. **Age out** tracks whose last observation is older than [TrackerConfig.maxAge]. Ageing is
 *    driven purely by the supplied timestamps, so an empty frame still ages every track and an
 *    expired track is gone for good - its id is never reused.
 * 2. **Predict** where each surviving track should be at the current timestamp, using its own
 *    motion history rather than an assumed frame rate.
 * 3. **Score** every detection against every track, taking the better of the score against the last
 *    observation and the score against the prediction. A stationary person keeps matching on the
 *    observation, a moving person keeps matching on the prediction.
 * 4. **Match** one-to-one with [OneToOneAssigner]. Scores below [TrackerConfig.minSimilarity] can
 *    never produce a link, so a detection that only weakly resembles a track stays unmatched
 *    instead of stealing that identity.
 * 5. **Update** matched tracks, spawn tracks for unmatched detections, and count a miss for every
 *    track that went unseen.
 *
 * The engine has no Android dependencies, which is what makes crossing, occlusion and expiry
 * scenarios testable on the JVM.
 */
class PoseTrackerEngine(
    val config: TrackerConfig,
    private val similarityFunction: PoseSimilarityFunction
) {

    private val liveTracks = mutableListOf<TrackRecord>()
    private val assigner = OneToOneAssigner()
    private var nextTrackId = 0

    private var similarityScratch: Array<FloatArray> = emptyArray()
    private var assignmentScratch = IntArray(0)

    /** Tracks that are currently alive, most recently seen first. */
    val tracks: List<TrackRecord> get() = liveTracks

    /** Maximum age of a track in microseconds before it is deleted. */
    val maxAgeMicros: Long get() = config.maxAge.toLong() * 1000L

    /**
     * Advances the tracker by one frame.
     *
     * @param detections poses found in this frame, most confident first. May be empty, which still
     * ages the existing tracks.
     * @param timestampMicros a monotonic timestamp in microseconds.
     * @return one entry per detection holding the assigned track id, or
     * [OneToOneAssigner.UNASSIGNED] for detections beyond [TrackerConfig.maxDetections].
     */
    fun update(detections: List<PoseFeature>, timestampMicros: Long): IntArray {
        val ids = IntArray(detections.size) { OneToOneAssigner.UNASSIGNED }

        expireTracks(timestampMicros)

        // The matching problem is bounded so a pathological frame cannot stall the pipeline.
        val detectionCount = min(detections.size, config.maxDetections)
        val trackCount = liveTracks.size

        if (detectionCount > 0 && trackCount > 0) {
            val similarity = obtainSimilarityMatrix(detectionCount, trackCount)
            for (d in 0 until detectionCount) {
                val detection = detections[d]
                val row = similarity[d]
                for (t in 0 until trackCount) {
                    row[t] = score(detection, liveTracks[t], timestampMicros)
                }
            }
            val assignment = obtainAssignmentScratch(detectionCount)
            assigner.assign(similarity, config.minSimilarity, assignment)

            val matchedTracks = BooleanArray(trackCount)
            for (d in 0 until detectionCount) {
                val t = assignment[d]
                if (t == OneToOneAssigner.UNASSIGNED) continue
                val track = liveTracks[t]
                track.observe(detections[d], timestampMicros)
                matchedTracks[t] = true
                ids[d] = track.id
            }
            for (t in 0 until trackCount) {
                if (!matchedTracks[t]) liveTracks[t].miss()
            }
        } else {
            for (track in liveTracks) track.miss()
        }

        for (d in 0 until detectionCount) {
            if (ids[d] != OneToOneAssigner.UNASSIGNED) continue
            val track = TrackRecord(
                id = ++nextTrackId,
                feature = detections[d],
                timestampMicros = timestampMicros,
                historySize = config.motionHistorySize,
                maxExtrapolationMicros = config.maxExtrapolationMillis.toLong() * 1000L
            )
            liveTracks.add(track)
            ids[d] = track.id
        }

        trimTracks()
        return ids
    }

    /**
     * Pairwise scores between [detections] and the live tracks, as used by [update].
     *
     * Exposed so the Android trackers can keep publishing their `computeSimilarity` contract.
     */
    fun computeSimilarity(detections: List<PoseFeature>, timestampMicros: Long): List<List<Float>> {
        if (detections.isEmpty()) return emptyList()
        return detections.map { detection ->
            liveTracks.map { track -> score(detection, track, timestampMicros) }
        }
    }

    /** Forgets every track. Ids continue from where they left off so no identity is ever reused. */
    fun reset() {
        liveTracks.clear()
    }

    /**
     * Best of the observed and the predicted similarity.
     *
     * Using the maximum keeps a stationary person matching on their last observation while a moving
     * person is matched against the extrapolated pose. Because the final decision is a global
     * one-to-one assignment, the extra candidate cannot cause an identity swap on its own: the
     * swapped pairing simply scores lower overall.
     */
    private fun score(detection: PoseFeature, track: TrackRecord, timestampMicros: Long): Float {
        val observed = similarityFunction.similarity(detection, track.feature)
        val predicted = track.predict(timestampMicros)
            ?.let { similarityFunction.similarity(detection, it) }
            ?: return observed
        return max(observed, predicted)
    }

    private fun expireTracks(timestampMicros: Long) {
        if (liveTracks.isEmpty()) return
        val limit = maxAgeMicros
        val iterator = liveTracks.iterator()
        while (iterator.hasNext()) {
            if (iterator.next().ageMicros(timestampMicros) > limit) iterator.remove()
        }
    }

    /**
     * Keeps the newest [TrackerConfig.maxTracks] tracks.
     *
     * The sort is stable, so tracks updated in the same frame keep the order in which they were
     * created and the tracker stays reproducible.
     */
    private fun trimTracks() {
        liveTracks.sortByDescending { it.lastTimestamp }
        while (liveTracks.size > config.maxTracks) {
            liveTracks.removeAt(liveTracks.size - 1)
        }
    }

    private fun obtainSimilarityMatrix(rows: Int, cols: Int): Array<FloatArray> {
        val current = similarityScratch
        if (current.size == rows && (rows == 0 || current[0].size == cols)) return current
        val created = Array(rows) { FloatArray(cols) }
        similarityScratch = created
        return created
    }

    private fun obtainAssignmentScratch(rows: Int): IntArray {
        if (assignmentScratch.size < rows) assignmentScratch = IntArray(rows)
        return assignmentScratch
    }
}
