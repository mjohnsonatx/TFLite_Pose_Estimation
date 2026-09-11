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
import org.tensorflow.lite.examples.poseestimation.data.Person

/**
 * Android facing tracker.
 *
 * The matching, motion and ageing logic lives in the framework independent [PoseTrackerEngine];
 * this class only translates between `Person` and [PoseFeature] and publishes the [Track] view the
 * UI and the instrumentation tests expect.
 *
 * The public contract - [apply], [tracks], [reset], [computeSimilarity] and [config] - is unchanged.
 * What changed is the behaviour behind it: detections and tracks are now matched one-to-one against
 * a motion prediction instead of greedily against the last observation.
 */
abstract class AbstractTracker(val config: TrackerConfig) {

    /** Similarity measure used to compare a detection with a track. */
    protected abstract val similarityFunction: PoseSimilarityFunction

    private val engine: PoseTrackerEngine by lazy { PoseTrackerEngine(config, similarityFunction) }

    /** Last `Person` observed for each live track id, used to project [tracks]. */
    private val personByTrackId = mutableMapOf<Int, Person>()

    /**
     * Live tracks, most recently observed first.
     *
     * Rebuilt on access from the engine state so callers can never mutate the tracker internals.
     */
    val tracks: List<Track>
        get() = engine.tracks.mapNotNull { record ->
            personByTrackId[record.id]?.let { Track(it, record.lastTimestamp) }
        }

    /**
     * Computes pairwise similarity scores between detections and tracks, based
     * on detected features.
     * @param persons A list of detected person.
     * @returns A list of shape [num_det, num_tracks] with pairwise
     * similarity scores between detections and tracks.
     */
    open fun computeSimilarity(persons: List<Person>): List<List<Float>> {
        if (persons.isEmpty()) return emptyList()
        val features = persons.map { it.toPoseFeature() }
        return features.map { detection ->
            engine.tracks.map { track -> similarityFunction.similarity(detection, track.feature) }
        }
    }

    /**
     * Tracks person instances across frames based on detections.
     *
     * The supplied `persons` are updated in place with their track id and returned, exactly as
     * before. An empty list is valid and still ages the existing tracks, so somebody who walks out
     * of frame and returns within [TrackerConfig.maxAge] can resume their identity.
     *
     * @param persons A list of person, most confident first.
     * @param timestamp The current timestamp in microseconds. Must come from a monotonic clock.
     * @return An updated list of persons with tracking id.
     */
    fun apply(persons: List<Person>, timestamp: Long): List<Person> {
        val ids = engine.update(persons.map { it.toPoseFeature() }, timestamp)
        for (index in persons.indices) {
            val id = ids[index]
            if (id == OneToOneAssigner.UNASSIGNED) continue
            persons[index].id = id
            personByTrackId[id] = Person(
                id = id,
                keyPoints = persons[index].keyPoints,
                boundingBox = persons[index].boundingBox,
                score = persons[index].score
            )
        }
        pruneSnapshots()
        return persons
    }

    /**
     * Clear all track in list of tracks
     */
    fun reset() {
        engine.reset()
        personByTrackId.clear()
    }

    /** Drops snapshots of tracks the engine has expired, so nothing is retained after `maxAge`. */
    private fun pruneSnapshots() {
        if (personByTrackId.isEmpty()) return
        val live = engine.tracks.mapTo(HashSet(engine.tracks.size)) { it.id }
        personByTrackId.keys.retainAll(live)
    }
}

/** Flattens a `Person` into the framework independent representation the engine works on. */
internal fun Person.toPoseFeature(): PoseFeature {
    val values = FloatArray(keyPoints.size * PoseFeature.VALUES_PER_KEY_POINT)
    keyPoints.forEachIndexed { index, keyPoint ->
        val base = index * PoseFeature.VALUES_PER_KEY_POINT
        values[base] = keyPoint.coordinate.x
        values[base + 1] = keyPoint.coordinate.y
        values[base + 2] = keyPoint.score
    }
    val box: FloatArray? = boundingBox?.let {
        floatArrayOf(it.left, it.top, it.right, it.bottom)
    }
    return PoseFeature(values, box, score)
}

/** Convenience for tests and callers that only need the box similarity of a `RectF`. */
internal fun RectF.toPoseFeature(score: Float = 1f): PoseFeature =
    PoseFeature.ofBox(left, top, right, bottom, score)
