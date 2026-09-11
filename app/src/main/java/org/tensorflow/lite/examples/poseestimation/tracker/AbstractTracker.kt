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
import org.tensorflow.lite.examples.poseestimation.data.KeyPoint
import org.tensorflow.lite.examples.poseestimation.data.Person

abstract class AbstractTracker(val config: TrackerConfig) {

    private val maxAge = config.maxAge * 1000L // convert milliseconds to microseconds
    private var nextTrackId = 0
    var tracks = mutableListOf<Track>()
        private set

    /**
     * Computes pairwise similarity scores between detections and tracks, based
     * on detected features.
     * @param persons A list of detected person.
     * @return A list of shape [num_det, num_tracks] with pairwise similarity scores.
     */
    abstract fun computeSimilarity(persons: List<Person>): List<List<Float>>

    /**
     * Computes pairwise similarity scores with timestamp for motion-predicted tracking.
     */
    open fun computeSimilarity(persons: List<Person>, timestamp: Long): List<List<Float>> {
        return computeSimilarity(persons)
    }

    /**
     * Tracks person instances across frames based on detections.
     * @param persons A list of person
     * @param timestamp The current timestamp in microseconds
     * @return An updated list of persons with tracking id.
     */
    fun apply(persons: List<Person>, timestamp: Long): List<Person> {
        tracks = filterOldTrack(timestamp).toMutableList()
        val simMatrix = computeSimilarity(persons, timestamp)
        assignTrack(persons, simMatrix, timestamp)
        tracks = updateTrack().toMutableList()
        return persons
    }

    /**
     * Clear all tracks in list of tracks
     */
    fun reset() {
        tracks.clear()
        nextTrackId = 0
    }

    /**
     * Return the next track id
     */
    private fun nextTrackID() = ++nextTrackId

    private data class MatchCandidate(
        val detectionIndex: Int,
        val trackIndex: Int,
        val similarity: Float,
        val trackId: Int
    )

    /**
     * Performs a one-to-one greedy assignment to link detections with tracks.
     * Ties are broken deterministically by detection index and track index.
     */
    private fun assignTrack(persons: List<Person>, simMatrix: List<List<Float>>, timestamp: Long) {
        if (persons.isEmpty()) {
            return
        }

        if (tracks.isEmpty()) {
            for (detectionIndex in persons.indices) {
                val newTrack = createTrack(persons[detectionIndex], timestamp = timestamp)
                tracks.add(newTrack)
                persons[detectionIndex].id = newTrack.person.id
            }
            return
        }

        require(simMatrix.size == persons.size && simMatrix.all { it.size == tracks.size }) {
            "Size of person array and similarity matrix does not match."
        }

        val candidates = mutableListOf<MatchCandidate>()
        for (d in persons.indices) {
            for (t in tracks.indices) {
                val similarity = simMatrix[d][t]
                if (similarity >= config.minSimilarity) {
                    candidates.add(MatchCandidate(d, t, similarity, tracks[t].person.id))
                }
            }
        }

        candidates.sortWith(
            compareByDescending<MatchCandidate> { it.similarity }
                .thenBy { it.detectionIndex }
                .thenBy { it.trackIndex }
        )

        val matchedDetections = BooleanArray(persons.size)
        val matchedTracks = BooleanArray(tracks.size)

        for (candidate in candidates) {
            val d = candidate.detectionIndex
            val t = candidate.trackIndex
            if (!matchedDetections[d] && !matchedTracks[t]) {
                matchedDetections[d] = true
                matchedTracks[t] = true
                val linkedTrack = tracks[t]
                tracks[t] = createTrack(persons[d], linkedTrack.person.id, timestamp, linkedTrack)
                persons[d].id = linkedTrack.person.id
            }
        }

        for (d in persons.indices) {
            if (!matchedDetections[d]) {
                val newTrack = createTrack(persons[d], timestamp = timestamp)
                tracks.add(newTrack)
                persons[d].id = newTrack.person.id
            }
        }
    }

    /**
     * Filters tracks based on their age.
     * @param timestamp The timestamp in microseconds
     */
    private fun filterOldTrack(timestamp: Long): List<Track> {
        return tracks.filter {
            timestamp - it.lastTimestamp <= maxAge
        }
    }

    /**
     * Sort the track list by timestamp (newer first)
     * and return the track list with size equal to config.maxTracks
     */
    private fun updateTrack(): List<Track> {
        tracks.sortByDescending { it.lastTimestamp }
        return tracks.take(config.maxTracks)
    }

    private fun keyPointsCenter(keyPoints: List<KeyPoint>): PointF? {
        val valid = keyPoints.filter { it.score >= (config.keyPointsTrackerParams?.keypointThreshold ?: 0.2f) }
        if (valid.isEmpty()) return null
        val avgX = valid.map { it.coordinate.x }.average().toFloat()
        val avgY = valid.map { it.coordinate.y }.average().toFloat()
        return PointF(avgX, avgY)
    }

    /**
     * Create a new track from person's information, updating velocity if linked to a previous track.
     */
    private fun createTrack(
        person: Person,
        id: Int? = null,
        timestamp: Long,
        prevTrack: Track? = null
    ): Track {
        var vx = 0f
        var vy = 0f

        if (prevTrack != null) {
            val dt = timestamp - prevTrack.lastTimestamp
            if (dt > 0) {
                if (person.boundingBox != null && prevTrack.person.boundingBox != null) {
                    val currCx = (person.boundingBox.left + person.boundingBox.right) / 2f
                    val currCy = (person.boundingBox.top + person.boundingBox.bottom) / 2f
                    val prevCx = (prevTrack.person.boundingBox.left + prevTrack.person.boundingBox.right) / 2f
                    val prevCy = (prevTrack.person.boundingBox.top + prevTrack.person.boundingBox.bottom) / 2f
                    val instVx = (currCx - prevCx) / dt
                    val instVy = (currCy - prevCy) / dt
                    vx = if (prevTrack.vx == 0f && prevTrack.vy == 0f) instVx else 0.7f * instVx + 0.3f * prevTrack.vx
                    vy = if (prevTrack.vx == 0f && prevTrack.vy == 0f) instVy else 0.7f * instVy + 0.3f * prevTrack.vy
                } else {
                    val currCenter = keyPointsCenter(person.keyPoints)
                    val prevCenter = keyPointsCenter(prevTrack.person.keyPoints)
                    if (currCenter != null && prevCenter != null) {
                        val instVx = (currCenter.x - prevCenter.x) / dt
                        val instVy = (currCenter.y - prevCenter.y) / dt
                        vx = if (prevTrack.vx == 0f && prevTrack.vy == 0f) instVx else 0.7f * instVx + 0.3f * prevTrack.vx
                        vy = if (prevTrack.vx == 0f && prevTrack.vy == 0f) instVy else 0.7f * instVy + 0.3f * prevTrack.vy
                    }
                }
            }
        }

        return Track(
            person = Person(
                id = id ?: nextTrackID(),
                keyPoints = person.keyPoints,
                boundingBox = person.boundingBox,
                score = person.score
            ),
            lastTimestamp = timestamp,
            vx = vx,
            vy = vy,
            lastKeyPoints = person.keyPoints,
            prevTimestamp = prevTrack?.lastTimestamp ?: timestamp,
            prevBoundingBox = prevTrack?.person?.boundingBox ?: person.boundingBox
        )
    }
}
