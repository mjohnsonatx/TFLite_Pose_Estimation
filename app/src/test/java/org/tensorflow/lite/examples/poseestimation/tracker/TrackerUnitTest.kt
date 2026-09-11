package org.tensorflow.lite.examples.poseestimation.tracker

import android.graphics.PointF
import android.graphics.RectF
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.tensorflow.lite.examples.poseestimation.data.BodyPart
import org.tensorflow.lite.examples.poseestimation.data.KeyPoint
import org.tensorflow.lite.examples.poseestimation.data.Person

class TrackerUnitTest {

    private lateinit var bbTracker: BoundingBoxTracker
    private lateinit var kpTracker: KeyPointsTracker

    @Before
    fun setup() {
        bbTracker = BoundingBoxTracker(
            TrackerConfig(
                maxTracks = 18,
                maxAge = 1000,
                minSimilarity = 0.3f
            )
        )
        kpTracker = KeyPointsTracker(
            TrackerConfig(
                maxTracks = 18,
                maxAge = 1000,
                minSimilarity = 0.3f,
                keyPointsTrackerParams = KeyPointsTrackerParams(
                    keypointThreshold = 0.2f,
                    keypointFalloff = listOf(0.1f, 0.1f, 0.1f, 0.1f),
                    minNumKeyPoints = 2
                )
            )
        )
    }

    @Test
    fun testCrossingTrajectories() {
        // Person 1 starts on left moving right (x: 0.10 -> 0.15 -> 0.35 -> 0.50 -> 0.70)
        // Person 2 starts on right moving left (x: 0.90 -> 0.85 -> 0.65 -> 0.50 -> 0.30)
        val t0 = 0L
        val p1_0 = Person(keyPoints = emptyList(), boundingBox = RectF(0.05f, 0.1f, 0.15f, 0.3f), score = 0.9f)
        val p2_0 = Person(keyPoints = emptyList(), boundingBox = RectF(0.85f, 0.1f, 0.95f, 0.3f), score = 0.9f)
        val f0 = bbTracker.apply(listOf(p1_0, p2_0), t0)
        val id1 = f0[0].id
        val id2 = f0[1].id
        assertNotEquals(id1, id2)

        // t1: 30ms later (establish initial velocity)
        val t1 = 30_000L
        val p1_1 = Person(keyPoints = emptyList(), boundingBox = RectF(0.10f, 0.1f, 0.20f, 0.3f), score = 0.9f)
        val p2_1 = Person(keyPoints = emptyList(), boundingBox = RectF(0.80f, 0.1f, 0.90f, 0.3f), score = 0.9f)
        val f1 = bbTracker.apply(listOf(p1_1, p2_1), t1)
        assertEquals(id1, f1[0].id)
        assertEquals(id2, f1[1].id)

        // t2: 150ms later (Moving towards center)
        val t2 = 150_000L
        val p1_2 = Person(keyPoints = emptyList(), boundingBox = RectF(0.30f, 0.1f, 0.40f, 0.3f), score = 0.9f)
        val p2_2 = Person(keyPoints = emptyList(), boundingBox = RectF(0.60f, 0.1f, 0.70f, 0.3f), score = 0.9f)
        val f2 = bbTracker.apply(listOf(p1_2, p2_2), t2)
        assertEquals(id1, f2[0].id)
        assertEquals(id2, f2[1].id)

        // t3: 250ms later (Crossing / overlapping near center x=0.5)
        val t3 = 250_000L
        val p1_3 = Person(keyPoints = emptyList(), boundingBox = RectF(0.46f, 0.1f, 0.56f, 0.3f), score = 0.9f)
        val p2_3 = Person(keyPoints = emptyList(), boundingBox = RectF(0.44f, 0.1f, 0.54f, 0.3f), score = 0.9f)
        val f3 = bbTracker.apply(listOf(p1_3, p2_3), t3)
        assertEquals(id1, f3[0].id)
        assertEquals(id2, f3[1].id)

        // t4: 370ms later (Crossed over: person 1 is now on right, person 2 is on left)
        val t4 = 370_000L
        val p1_4 = Person(keyPoints = emptyList(), boundingBox = RectF(0.65f, 0.1f, 0.75f, 0.3f), score = 0.9f)
        val p2_4 = Person(keyPoints = emptyList(), boundingBox = RectF(0.25f, 0.1f, 0.35f, 0.3f), score = 0.9f)
        val f4 = bbTracker.apply(listOf(p1_4, p2_4), t4)

        // Person 1 continuing right should still have id1
        assertEquals("Person moving right should maintain id1", id1, f4[0].id)
        // Person 2 continuing left should still have id2
        assertEquals("Person moving left should maintain id2", id2, f4[1].id)
    }

    @Test
    fun testShortOcclusionAndReappearance() {
        val t0 = 0L
        val person = Person(keyPoints = emptyList(), boundingBox = RectF(0.2f, 0.2f, 0.4f, 0.6f), score = 0.9f)
        val f0 = bbTracker.apply(listOf(person), t0)
        val originalId = f0[0].id

        // Occlusion for 2 empty frames (200ms and 400ms)
        val f1 = bbTracker.apply(emptyList(), 200_000L)
        assertTrue(f1.isEmpty())
        assertEquals(1, bbTracker.tracks.size)

        val f2 = bbTracker.apply(emptyList(), 400_000L)
        assertTrue(f2.isEmpty())
        assertEquals(1, bbTracker.tracks.size)

        // Re-appears at 500ms (< maxAge of 1000ms)
        val reappearedPerson = Person(keyPoints = emptyList(), boundingBox = RectF(0.22f, 0.2f, 0.42f, 0.6f), score = 0.9f)
        val f3 = bbTracker.apply(listOf(reappearedPerson), 500_000L)
        assertEquals(1, f3.size)
        assertEquals("Reappeared person within maxAge should resume id", originalId, f3[0].id)
    }

    @Test
    fun testTrackExpirationCannotBeRevived() {
        val t0 = 0L
        val person = Person(keyPoints = emptyList(), boundingBox = RectF(0.2f, 0.2f, 0.4f, 0.6f), score = 0.9f)
        val f0 = bbTracker.apply(listOf(person), t0)
        val originalId = f0[0].id

        // Person disappears for 1100ms (> maxAge 1000ms)
        val tExpired = 1_100_000L
        val f1 = bbTracker.apply(emptyList(), tExpired)
        assertTrue(f1.isEmpty())
        assertTrue("Expired track must be removed", bbTracker.tracks.isEmpty())

        // Person re-appears with identical bounding box at 1200ms
        val returnedPerson = Person(keyPoints = emptyList(), boundingBox = RectF(0.2f, 0.2f, 0.4f, 0.6f), score = 0.9f)
        val f2 = bbTracker.apply(listOf(returnedPerson), 1_200_000L)
        assertEquals(1, f2.size)
        assertNotEquals("Expired track cannot be revived; new id should be created", originalId, f2[0].id)
    }

    @Test
    fun testReorderedDetections() {
        val t0 = 0L
        val p1 = Person(keyPoints = emptyList(), boundingBox = RectF(0.1f, 0.1f, 0.3f, 0.3f), score = 0.9f)
        val p2 = Person(keyPoints = emptyList(), boundingBox = RectF(0.7f, 0.7f, 0.9f, 0.9f), score = 0.9f)
        val f0 = bbTracker.apply(listOf(p1, p2), t0)
        val id1 = f0[0].id
        val id2 = f0[1].id

        // Frame 1: pass detections in reverse order [p2, p1]
        val t1 = 50_000L
        val p1_next = Person(keyPoints = emptyList(), boundingBox = RectF(0.12f, 0.12f, 0.32f, 0.32f), score = 0.9f)
        val p2_next = Person(keyPoints = emptyList(), boundingBox = RectF(0.72f, 0.72f, 0.92f, 0.92f), score = 0.9f)
        val f1 = bbTracker.apply(listOf(p2_next, p1_next), t1)

        assertEquals(id2, f1[0].id)
        assertEquals(id1, f1[1].id)
    }

    @Test
    fun testMoreTracksThanDetections() {
        val t0 = 0L
        val p1 = Person(keyPoints = emptyList(), boundingBox = RectF(0.1f, 0.1f, 0.2f, 0.2f), score = 0.9f)
        val p2 = Person(keyPoints = emptyList(), boundingBox = RectF(0.4f, 0.4f, 0.5f, 0.5f), score = 0.9f)
        val p3 = Person(keyPoints = emptyList(), boundingBox = RectF(0.7f, 0.7f, 0.8f, 0.8f), score = 0.9f)
        val f0 = bbTracker.apply(listOf(p1, p2, p3), t0)
        val id1 = f0[0].id
        val id2 = f0[1].id
        val id3 = f0[2].id

        // Frame 1: Only person 2 is detected
        val t1 = 30_000L
        val p2_next = Person(keyPoints = emptyList(), boundingBox = RectF(0.41f, 0.41f, 0.51f, 0.51f), score = 0.9f)
        val f1 = bbTracker.apply(listOf(p2_next), t1)

        assertEquals(1, f1.size)
        assertEquals(id2, f1[0].id)
        assertEquals(3, bbTracker.tracks.size)
    }

    @Test
    fun testMoreDetectionsThanTracks() {
        val t0 = 0L
        val p1 = Person(keyPoints = emptyList(), boundingBox = RectF(0.1f, 0.1f, 0.2f, 0.2f), score = 0.9f)
        val f0 = bbTracker.apply(listOf(p1), t0)
        val id1 = f0[0].id

        // Frame 1: Person 1 + 3 new people
        val t1 = 30_000L
        val p1_next = Person(keyPoints = emptyList(), boundingBox = RectF(0.11f, 0.11f, 0.21f, 0.21f), score = 0.9f)
        val p2 = Person(keyPoints = emptyList(), boundingBox = RectF(0.3f, 0.3f, 0.4f, 0.4f), score = 0.9f)
        val p3 = Person(keyPoints = emptyList(), boundingBox = RectF(0.5f, 0.5f, 0.6f, 0.6f), score = 0.9f)
        val p4 = Person(keyPoints = emptyList(), boundingBox = RectF(0.7f, 0.7f, 0.8f, 0.8f), score = 0.9f)

        val f1 = bbTracker.apply(listOf(p1_next, p2, p3, p4), t1)
        assertEquals(4, f1.size)
        assertEquals(id1, f1[0].id)
        assertNotEquals(id1, f1[1].id)
        assertNotEquals(id1, f1[2].id)
        assertNotEquals(id1, f1[3].id)
        val ids = f1.map { it.id }.toSet()
        assertEquals(4, ids.size)
    }

    @Test
    fun testBelowThresholdResults() {
        val t0 = 0L
        val p1 = Person(keyPoints = emptyList(), boundingBox = RectF(0.1f, 0.1f, 0.3f, 0.3f), score = 0.9f)
        val f0 = bbTracker.apply(listOf(p1), t0)
        val id1 = f0[0].id

        // Frame 1: Person at completely different non-overlapping position (IoU = 0 < minSimilarity 0.3)
        val t1 = 30_000L
        val pFar = Person(keyPoints = emptyList(), boundingBox = RectF(0.7f, 0.7f, 0.9f, 0.9f), score = 0.9f)
        val f1 = bbTracker.apply(listOf(pFar), t1)

        assertEquals(1, f1.size)
        assertNotEquals("Below threshold similarity must not be assigned to existing track", id1, f1[0].id)
    }

    @Test
    fun testDeterministicTieBreaking() {
        val t0 = 0L
        // Create 2 identical tracks
        val p1 = Person(keyPoints = emptyList(), boundingBox = RectF(0.2f, 0.2f, 0.4f, 0.4f), score = 0.9f)
        val p2 = Person(keyPoints = emptyList(), boundingBox = RectF(0.2f, 0.2f, 0.4f, 0.4f), score = 0.9f)
        val f0 = bbTracker.apply(listOf(p1, p2), t0)
        val id1 = f0[0].id
        val id2 = f0[1].id

        // Run matching multiple times with identical detections - tie breaking must produce deterministic result
        val pNext1 = Person(keyPoints = emptyList(), boundingBox = RectF(0.2f, 0.2f, 0.4f, 0.4f), score = 0.9f)
        val pNext2 = Person(keyPoints = emptyList(), boundingBox = RectF(0.2f, 0.2f, 0.4f, 0.4f), score = 0.9f)
        val f1 = bbTracker.apply(listOf(pNext1, pNext2), 50_000L)

        assertEquals(id1, f1[0].id)
        assertEquals(id2, f1[1].id)
    }

    @Test
    fun testUpTo6DetectionsAnd18Tracks() {
        val tracker = BoundingBoxTracker(TrackerConfig(maxTracks = 18, maxAge = 5000, minSimilarity = 0.1f))

        // Create 18 tracks across multiple frames
        var time = 0L
        for (i in 0 until 3) {
            val detections = (0 until 6).map { j ->
                val x = (i * 6 + j) * 0.05f
                Person(keyPoints = emptyList(), boundingBox = RectF(x, 0.1f, x + 0.04f, 0.3f), score = 0.9f)
            }
            tracker.apply(detections, time)
            time += 100_000L
        }

        assertEquals(18, tracker.tracks.size)

        // Now test 6 detections matching 6 of the 18 tracks
        val activeDetections = (0 until 6).map { j ->
            val x = j * 0.05f + 0.005f
            Person(keyPoints = emptyList(), boundingBox = RectF(x, 0.1f, x + 0.04f, 0.3f), score = 0.9f)
        }
        val result = tracker.apply(activeDetections, time)
        assertEquals(6, result.size)
        assertEquals(18, tracker.tracks.size)
    }

    @Test
    fun testKeyPointsTrackerOcclusionAndMotion() {
        val t0 = 0L
        val kps0 = listOf(
            KeyPoint(BodyPart.NOSE, PointF(0.2f, 0.2f), 0.9f),
            KeyPoint(BodyPart.LEFT_EYE, PointF(0.25f, 0.25f), 0.9f),
            KeyPoint(BodyPart.LEFT_SHOULDER, PointF(0.3f, 0.4f), 0.9f),
            KeyPoint(BodyPart.RIGHT_SHOULDER, PointF(0.4f, 0.4f), 0.9f)
        )
        val f0 = kpTracker.apply(listOf(Person(keyPoints = kps0, score = 0.9f)), t0)
        val id1 = f0[0].id

        // Advance 30ms with slight motion to establish velocity (dx = 0.03)
        val t1 = 30_000L
        val kps1 = listOf(
            KeyPoint(BodyPart.NOSE, PointF(0.23f, 0.2f), 0.9f),
            KeyPoint(BodyPart.LEFT_EYE, PointF(0.28f, 0.25f), 0.9f),
            KeyPoint(BodyPart.LEFT_SHOULDER, PointF(0.33f, 0.4f), 0.9f),
            KeyPoint(BodyPart.RIGHT_SHOULDER, PointF(0.43f, 0.4f), 0.9f)
        )
        val f1 = kpTracker.apply(listOf(Person(keyPoints = kps1, score = 0.9f)), t1)
        assertEquals(id1, f1[0].id)

        // Empty frame at 100ms (occlusion)
        kpTracker.apply(emptyList(), 100_000L)

        // Reappearance at 200ms following predicted trajectory (dx = 0.17 from t1)
        val t3 = 200_000L
        val kps3 = listOf(
            KeyPoint(BodyPart.NOSE, PointF(0.40f, 0.2f), 0.9f),
            KeyPoint(BodyPart.LEFT_EYE, PointF(0.45f, 0.25f), 0.9f),
            KeyPoint(BodyPart.LEFT_SHOULDER, PointF(0.50f, 0.4f), 0.9f),
            KeyPoint(BodyPart.RIGHT_SHOULDER, PointF(0.60f, 0.4f), 0.9f)
        )
        val f3 = kpTracker.apply(listOf(Person(keyPoints = kps3, score = 0.9f)), t3)
        assertEquals("KeyPoint tracker should resume identity on predicted path", id1, f3[0].id)
    }
}
