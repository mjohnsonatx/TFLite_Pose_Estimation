package org.tensorflow.lite.examples.poseestimation.camera

/**
 * Thread-safe single-slot frame queue that enforces a backpressure strategy:
 * - Keeps only the latest pending frame when processing is busy.
 * - Any displaced stale frame is immediately closed/disposed.
 * - On close/shutdown, any pending frame is closed.
 * - Properly handles exceptions, cancellation, and clean shutdown.
 */
class FrameBackpressureQueue<T : AutoCloseable> : AutoCloseable {

    private val lock = Any()
    private var pendingFrame: T? = null
    private var isProcessing: Boolean = false
    private var isClosed: Boolean = false

    var droppedFramesCount: Long = 0
        private set

    /**
     * Offers a new frame to the queue.
     * @param frame The incoming frame.
     * @return FrameAction indicating whether to start processing immediately or if the frame was enqueued / closed.
     */
    fun offer(frame: T): FrameAction {
        synchronized(lock) {
            if (isClosed) {
                try {
                    frame.close()
                } catch (_: Throwable) {
                }
                return FrameAction.DISCARDED_CLOSED
            }

            if (!isProcessing) {
                isProcessing = true
                return FrameAction.PROCESS_NOW
            }

            // A frame is currently being processed. Place in single-slot pending queue.
            val stale = pendingFrame
            pendingFrame = frame
            if (stale != null) {
                droppedFramesCount++
                try {
                    stale.close()
                } catch (_: Throwable) {
                }
                return FrameAction.REPLACED_PENDING
            }

            return FrameAction.ENQUEUED_PENDING
        }
    }

    /**
     * Called when processing of the current frame completes (or fails).
     * @return The next frame to process if available, or null if idle.
     */
    fun onFrameCompleted(): T? {
        synchronized(lock) {
            if (isClosed) {
                pendingFrame?.let {
                    try {
                        it.close()
                    } catch (_: Throwable) {
                    }
                }
                pendingFrame = null
                isProcessing = false
                return null
            }

            val next = pendingFrame
            if (next != null) {
                pendingFrame = null
                return next
            } else {
                isProcessing = false
                return null
            }
        }
    }

    fun isBusy(): Boolean = synchronized(lock) { isProcessing }

    fun hasPending(): Boolean = synchronized(lock) { pendingFrame != null }

    override fun close() {
        synchronized(lock) {
            isClosed = true
            pendingFrame?.let {
                try {
                    it.close()
                } catch (_: Throwable) {
                }
            }
            pendingFrame = null
            isProcessing = false
        }
    }
}

enum class FrameAction {
    PROCESS_NOW,
    ENQUEUED_PENDING,
    REPLACED_PENDING,
    DISCARDED_CLOSED
}
