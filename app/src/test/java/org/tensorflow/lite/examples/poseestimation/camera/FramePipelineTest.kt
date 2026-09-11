package org.tensorflow.lite.examples.poseestimation.camera

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger

class FramePipelineTest {

    private class TestFrame(val id: Int) : AutoCloseable {
        val isClosed = AtomicBoolean(false)
        val closeCount = AtomicInteger(0)

        override fun close() {
            isClosed.set(true)
            closeCount.incrementAndGet()
        }
    }

    @Test
    fun testBackpressureSinglePendingAndStaleDisposal() {
        val queue = FrameBackpressureQueue<TestFrame>()

        val frame1 = TestFrame(1)
        val action1 = queue.offer(frame1)
        assertEquals(FrameAction.PROCESS_NOW, action1)
        assertTrue(queue.isBusy())
        assertFalse(queue.hasPending())

        // Frame 2 arrives while Frame 1 is being processed
        val frame2 = TestFrame(2)
        val action2 = queue.offer(frame2)
        assertEquals(FrameAction.ENQUEUED_PENDING, action2)
        assertTrue(queue.hasPending())
        assertFalse(frame2.isClosed.get())

        // Frame 3 arrives while Frame 1 is still being processed -> replaces Frame 2
        val frame3 = TestFrame(3)
        val action3 = queue.offer(frame3)
        assertEquals(FrameAction.REPLACED_PENDING, action3)
        assertTrue(frame2.isClosed.get()) // Frame 2 must be closed immediately!
        assertEquals(1, frame2.closeCount.get())
        assertFalse(frame3.isClosed.get())
        assertEquals(1, queue.droppedFramesCount)

        // Frame 1 completes processing
        frame1.close()
        assertTrue(frame1.isClosed.get())

        val nextFrame = queue.onFrameCompleted()
        assertEquals(frame3, nextFrame)
        assertFalse(queue.hasPending())
        assertTrue(queue.isBusy())

        // Frame 3 completes processing
        frame3.close()
        val idle = queue.onFrameCompleted()
        assertNull(idle)
        assertFalse(queue.isBusy())
    }

    @Test
    fun testQueueShutdownDisposesPendingFrame() {
        val queue = FrameBackpressureQueue<TestFrame>()

        val frame1 = TestFrame(1)
        queue.offer(frame1)

        val frame2 = TestFrame(2)
        queue.offer(frame2)
        assertFalse(frame2.isClosed.get())

        // Queue is closed / activity destroyed
        queue.close()
        assertTrue("Pending frame must be closed on queue close", frame2.isClosed.get())

        // Offering after close must immediately close incoming frame
        val frame3 = TestFrame(3)
        val action = queue.offer(frame3)
        assertEquals(FrameAction.DISCARDED_CLOSED, action)
        assertTrue("Frame offered after close must be closed", frame3.isClosed.get())

        val next = queue.onFrameCompleted()
        assertNull(next)
    }

    @Test
    fun testExceptionDuringProcessingSafelyDisposesFrames() {
        val queue = FrameBackpressureQueue<TestFrame>()
        val frame1 = TestFrame(1)
        val frame2 = TestFrame(2)

        queue.offer(frame1)
        queue.offer(frame2)

        try {
            // Simulate exception during processing of frame1
            throw RuntimeException("Inference failed")
        } catch (_: Exception) {
            // Error handled
        } finally {
            frame1.close()
        }
        assertTrue(frame1.isClosed.get())

        val next = queue.onFrameCompleted()
        assertEquals(frame2, next)
        frame2.close()
        assertTrue(frame2.isClosed.get())
        assertNull(queue.onFrameCompleted())
    }

    @Test
    fun testResourceManagerExclusivityAndLifecycle() {
        val manager = FrameResourceManager()

        assertTrue(manager.acquire())
        assertFalse("Cannot acquire already busy resource", manager.acquire())
        assertTrue(manager.isInUse())

        manager.prepare(640, 480, 90)
        assertNotNull(manager.rotatedBitmap)
        assertNotNull(manager.visualizationBitmap)
        assertEquals(480, manager.currentWidth)
        assertEquals(640, manager.currentHeight)

        val initialRotated = manager.rotatedBitmap
        val initialVis = manager.visualizationBitmap

        // Calling prepare with same dimensions and rotation should not reallocate
        manager.prepare(640, 480, 90)
        assertTrue(initialRotated === manager.rotatedBitmap)
        assertTrue(initialVis === manager.visualizationBitmap)

        // Changing rotation reallocates
        manager.prepare(640, 480, 0)
        assertEquals(640, manager.currentWidth)
        assertEquals(480, manager.currentHeight)
        assertTrue(initialRotated!!.isRecycled)
        assertTrue(initialVis!!.isRecycled)

        manager.release()
        assertFalse(manager.isInUse())

        // Close / onDestroy
        manager.close()
        assertTrue(manager.isReleased())
        assertNull(manager.rotatedBitmap)
        assertNull(manager.visualizationBitmap)
        assertFalse("Cannot acquire after close", manager.acquire())
    }
}
