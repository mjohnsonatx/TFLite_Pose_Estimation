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

package org.tensorflow.lite.examples.poseestimation.imaging

/**
 * Single-owner guard around a mutable resource that is recycled between frames.
 *
 * Reusing a pixel array or a [android.graphics.Bitmap] across frames is only safe while exactly one
 * consumer touches it. This guard makes that contract explicit: a second caller that tries to use
 * the resource while an inference pass still holds it fails fast instead of silently corrupting the
 * frame that is being analysed.
 *
 * The guard is intentionally *not* reentrant. A nested acquire is a bug, not something to wait for.
 */
class ResourceGuard(private val name: String = "resource") {

    private val lock = Object()
    private var busy = false
    private var closed = false

    /** True while a caller holds the guard. */
    val isBusy: Boolean
        get() = synchronized(lock) { busy }

    /** True once [close] has been called. A closed guard can never be acquired again. */
    val isClosed: Boolean
        get() = synchronized(lock) { closed }

    /**
     * Acquires the guard.
     *
     * @throws IllegalStateException if the guard is closed or already held.
     */
    fun acquire() {
        synchronized(lock) {
            check(!closed) { "$name has been closed" }
            check(!busy) { "$name is already in use by another frame" }
            busy = true
        }
    }

    /** Acquires the guard, returning `false` instead of throwing when it is closed or held. */
    fun tryAcquire(): Boolean {
        synchronized(lock) {
            if (closed || busy) return false
            busy = true
            return true
        }
    }

    /**
     * Releases the guard.
     *
     * Releasing a guard that is not held is a programming error and throws, which surfaces the
     * double-release leaks that make reusable buffers so hard to debug.
     */
    fun release() {
        synchronized(lock) {
            check(busy) { "$name was released without being acquired" }
            busy = false
            lock.notifyAll()
        }
    }

    /**
     * Runs [block] while holding the guard, releasing it even when [block] throws.
     */
    inline fun <T> withResource(block: () -> T): T {
        acquire()
        try {
            return block()
        } finally {
            release()
        }
    }

    /**
     * Marks the guard as closed so no further work can be started.
     *
     * @return `true` when the resource is idle and may be recycled immediately, `false` when a
     * consumer still holds it. Callers that must recycle native memory should use [awaitIdle].
     */
    fun close(): Boolean {
        synchronized(lock) {
            closed = true
            lock.notifyAll()
            return !busy
        }
    }

    /**
     * Closes the guard and waits until the in-flight consumer, if any, has released it.
     *
     * @param timeoutMillis maximum time to wait; `0` waits forever.
     * @return `true` when the resource is idle and safe to recycle.
     */
    fun awaitIdle(timeoutMillis: Long = 0L): Boolean {
        require(timeoutMillis >= 0) { "timeoutMillis must not be negative" }
        val deadline = if (timeoutMillis == 0L) Long.MAX_VALUE else now() + timeoutMillis
        synchronized(lock) {
            closed = true
            lock.notifyAll()
            while (busy) {
                val remaining = deadline - now()
                if (remaining <= 0) return false
                try {
                    lock.wait(if (timeoutMillis == 0L) 0L else remaining)
                } catch (interrupted: InterruptedException) {
                    Thread.currentThread().interrupt()
                    return !busy
                }
            }
            return true
        }
    }

    private fun now(): Long = System.nanoTime() / 1_000_000L
}
