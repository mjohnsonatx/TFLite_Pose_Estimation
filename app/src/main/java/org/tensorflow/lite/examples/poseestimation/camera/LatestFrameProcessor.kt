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

package org.tensorflow.lite.examples.poseestimation.camera

import java.util.concurrent.Executor
import java.util.concurrent.RejectedExecutionException
import java.util.concurrent.atomic.AtomicLong

/**
 * Latest-frame-wins backpressure in front of a slow consumer.
 *
 * The processor holds a single pending slot plus the frame currently being processed. When
 * inference is slower than the capture interval the queued frame is replaced by the newer one, so
 * the model always sees the freshest image and the camera never blocks.
 *
 * Every frame handed to [submit] is released exactly once through `recycle`, no matter whether it
 * was dropped for staleness, processed successfully, cancelled, rejected after [close], or the
 * consumer threw. That guarantee is what stops `ImageReader` from starving after a few seconds.
 *
 * The class has no Android dependencies so the disposal contract can be verified by local JVM
 * unit tests with a deterministic [Executor].
 *
 * @param executor executes the drain loop; a single threaded executor is expected.
 * @param recycle releases a frame, e.g. `Image::close`.
 * @param onError reports consumer and executor failures.
 * @param process the (slow) consumer.
 */
class LatestFrameProcessor<T : Any>(
    private val executor: Executor,
    private val recycle: (T) -> Unit,
    private val onError: (Throwable) -> Unit = {},
    private val process: (T) -> Unit
) : AutoCloseable {

    private val lock = Object()
    private var pending: T? = null
    private var draining = false
    private var closed = false

    private val submitted = AtomicLong()
    private val dropped = AtomicLong()
    private val processed = AtomicLong()
    private val failed = AtomicLong()
    private val cancelled = AtomicLong()
    private val rejected = AtomicLong()
    private val recycled = AtomicLong()

    /** Frames accepted into the pending slot. */
    val submittedFrames: Long get() = submitted.get()

    /** Frames replaced in the pending slot because a newer frame arrived. */
    val droppedFrames: Long get() = dropped.get()

    /** Frames the consumer completed without throwing. */
    val processedFrames: Long get() = processed.get()

    /** Frames whose consumer threw. */
    val failedFrames: Long get() = failed.get()

    /** Frames discarded by [cancelPending] or [close]. */
    val cancelledFrames: Long get() = cancelled.get()

    /** Frames refused because the processor was closed or the executor rejected the drain. */
    val rejectedFrames: Long get() = rejected.get()

    /** Frames released back to their owner. Always equals the number of frames handed to [submit]. */
    val recycledFrames: Long get() = recycled.get()

    /** True when nothing is queued and no frame is being processed. */
    val isIdle: Boolean
        get() = synchronized(lock) { pending == null && !draining }

    /** True once [close] has been called. */
    val isClosed: Boolean
        get() = synchronized(lock) { closed }

    /**
     * Offers [frame] for processing, replacing any frame that is still waiting.
     *
     * @return `true` when the frame was queued, `false` when it was refused and already recycled.
     */
    fun submit(frame: T): Boolean {
        var replaced: T? = null
        var schedule = false
        var refuse = false
        synchronized(lock) {
            if (closed) {
                refuse = true
            } else {
                replaced = pending
                pending = frame
                if (!draining) {
                    draining = true
                    schedule = true
                }
            }
        }

        if (refuse) {
            rejected.incrementAndGet()
            release(frame)
            return false
        }

        submitted.incrementAndGet()
        replaced?.let {
            dropped.incrementAndGet()
            release(it)
        }
        if (schedule) {
            return scheduleDrain()
        }
        return true
    }

    /** Discards the queued frame, if any, without closing the processor. */
    fun cancelPending() {
        val orphan = synchronized(lock) {
            val frame = pending
            pending = null
            frame
        }
        orphan?.let {
            cancelled.incrementAndGet()
            release(it)
        }
    }

    /**
     * Waits until the pending slot is empty and the consumer is idle.
     *
     * @param timeoutMillis maximum time to wait; `0` waits forever.
     * @return `true` when the processor became idle.
     */
    fun awaitIdle(timeoutMillis: Long = 0L): Boolean {
        require(timeoutMillis >= 0) { "timeoutMillis must not be negative" }
        val deadline = if (timeoutMillis == 0L) Long.MAX_VALUE else nowMillis() + timeoutMillis
        synchronized(lock) {
            while (pending != null || draining) {
                val remaining = deadline - nowMillis()
                if (timeoutMillis != 0L && remaining <= 0) return false
                try {
                    lock.wait(if (timeoutMillis == 0L) 0L else remaining)
                } catch (interrupted: InterruptedException) {
                    Thread.currentThread().interrupt()
                    return pending == null && !draining
                }
            }
            return true
        }
    }

    /**
     * Stops accepting frames and releases the queued one.
     *
     * Frames already inside the consumer are still recycled by the drain loop. Safe to call twice.
     */
    override fun close() {
        val orphan = synchronized(lock) {
            if (closed) return
            closed = true
            val frame = pending
            pending = null
            lock.notifyAll()
            frame
        }
        orphan?.let {
            cancelled.incrementAndGet()
            release(it)
        }
    }

    private fun scheduleDrain(): Boolean {
        try {
            executor.execute(::drain)
            return true
        } catch (rejection: RejectedExecutionException) {
            val orphan = synchronized(lock) {
                draining = false
                val frame = pending
                pending = null
                lock.notifyAll()
                frame
            }
            orphan?.let {
                rejected.incrementAndGet()
                release(it)
            }
            reportError(rejection)
            return false
        }
    }

    private fun drain() {
        while (true) {
            val frame = synchronized(lock) {
                val next = pending
                pending = null
                if (next == null) {
                    draining = false
                    lock.notifyAll()
                }
                next
            } ?: return

            try {
                process(frame)
                processed.incrementAndGet()
            } catch (error: Throwable) {
                failed.incrementAndGet()
                reportError(error)
            } finally {
                release(frame)
            }
        }
    }

    private fun release(frame: T) {
        recycled.incrementAndGet()
        try {
            recycle(frame)
        } catch (error: Throwable) {
            reportError(error)
        }
    }

    private fun reportError(error: Throwable) {
        try {
            onError(error)
        } catch (ignored: Throwable) {
            // A failing error reporter must never break the frame pipeline.
        }
    }

    private fun nowMillis(): Long = System.nanoTime() / 1_000_000L
}
