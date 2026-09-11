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

import kotlin.math.abs

/**
 * Optimal one-to-one matching between detections and tracks.
 *
 * The previous tracker linked detections greedily: the first detection took the best track it could
 * find, which permanently swapped identities whenever two people passed each other. This solver
 * instead maximises the *total* similarity over the whole frame, so a locally attractive but
 * globally wrong pairing is rejected.
 *
 * It is the Jonker-Volgenant shortest augmenting path form of the Hungarian algorithm and runs in
 * `O(rows^2 * cols)`. For the supported problem size - up to
 * [TrackerConfig.maxDetections] detections against [TrackerConfig.maxTracks] tracks - that is a few
 * thousand operations per frame. No permutation or factorial search is involved and no third party
 * library is required.
 *
 * Behaviour worth relying on:
 * * Rectangular, empty and single-element matrices are all supported.
 * * Pairs scoring below `minSimilarity` are treated as forbidden. They are only ever produced when
 *   the solver is forced to, and are then reported as unmatched, so a weak score never steals an
 *   identity.
 * * The number of valid matches is maximised first, and the total similarity second.
 * * Ties are broken deterministically towards the lowest column index, and therefore towards the
 *   lowest track index for a given detection.
 *
 * Scratch storage grows monotonically and is reused between frames.
 */
class OneToOneAssigner {

    private var costs: Array<DoubleArray> = emptyArray()
    private var costRowCapacity = 0
    private var costColumnCapacity = 0
    private var potentialRow = DoubleArray(0)
    private var potentialCol = DoubleArray(0)
    private var columnToRow = IntArray(0)
    private var path = IntArray(0)
    private var minValue = DoubleArray(0)
    private var used = BooleanArray(0)

    /**
     * Matches the rows (detections) of [similarity] against its columns (tracks).
     *
     * @param similarity a `[detections][tracks]` matrix; every row must have the same length.
     * @param minSimilarity scores strictly below this value never produce a match.
     * @return a freshly allocated array with one entry per detection holding the matched track
     * index, or [UNASSIGNED].
     */
    fun assign(similarity: Array<FloatArray>, minSimilarity: Float): IntArray {
        val result = IntArray(similarity.size)
        assign(similarity, minSimilarity, result)
        return result
    }

    /**
     * Allocation-free variant of [assign] that fills [out].
     *
     * @param out must hold at least `similarity.size` entries; the first `similarity.size` are
     * overwritten.
     */
    fun assign(similarity: Array<FloatArray>, minSimilarity: Float, out: IntArray) {
        val rows = similarity.size
        require(out.size >= rows) {
            "Result array holds ${out.size} entries but $rows are required"
        }
        out.fill(UNASSIGNED, 0, rows)
        if (rows == 0) return

        val cols = similarity[0].size
        for (row in similarity) {
            require(row.size == cols) {
                "Similarity matrix is ragged: expected $cols columns but found ${row.size}"
            }
        }
        if (cols == 0) return

        // The solver requires at least as many columns as rows; transpose when that does not hold.
        val transposed = rows > cols
        val n = if (transposed) cols else rows
        val m = if (transposed) rows else cols

        ensureCapacity(n, m)

        // Any single forbidden pair must cost more than every feasible assignment combined, so the
        // solver maximises the number of valid matches before it maximises their similarity.
        var largest = 0.0
        for (i in 0 until rows) {
            for (j in 0 until cols) {
                val value = similarity[i][j]
                if (value.isFinite() && value >= minSimilarity) {
                    val magnitude = abs(value.toDouble())
                    if (magnitude > largest) largest = magnitude
                }
            }
        }
        val forbidden = largest * (n + 1) + 1.0

        for (i in 0 until n) {
            val costRow = costs[i]
            for (j in 0 until m) {
                val value = if (transposed) similarity[j][i] else similarity[i][j]
                costRow[j] = if (value.isFinite() && value >= minSimilarity) {
                    -value.toDouble()
                } else {
                    forbidden
                }
            }
        }

        solve(n, m)

        for (column in 1..m) {
            val row = columnToRow[column]
            if (row == 0) continue
            val detection = if (transposed) column - 1 else row - 1
            val track = if (transposed) row - 1 else column - 1
            // Forbidden pairs the solver was forced into are reported as unmatched.
            val score = similarity[detection][track]
            if (score.isFinite() && score >= minSimilarity) {
                out[detection] = track
            }
        }
    }

    /**
     * Hungarian algorithm on the 1-indexed `n x m` cost matrix in [costs] with `n <= m`.
     *
     * Fills [columnToRow] so that `columnToRow[j]` is the 1-indexed row matched with column `j`, or
     * `0` when the column is unmatched.
     */
    private fun solve(n: Int, m: Int) {
        java.util.Arrays.fill(potentialRow, 0, n + 1, 0.0)
        java.util.Arrays.fill(potentialCol, 0, m + 1, 0.0)
        java.util.Arrays.fill(columnToRow, 0, m + 1, 0)

        for (row in 1..n) {
            columnToRow[0] = row
            var column = 0
            java.util.Arrays.fill(minValue, 0, m + 1, Double.MAX_VALUE)
            java.util.Arrays.fill(used, 0, m + 1, false)
            do {
                used[column] = true
                val currentRow = columnToRow[column]
                var delta = Double.MAX_VALUE
                var nextColumn = 0
                // Scanning ascending with a strict comparison makes ties resolve to the lowest
                // column index, which is what keeps the whole tracker reproducible.
                for (candidate in 1..m) {
                    if (used[candidate]) continue
                    val reduced = costs[currentRow - 1][candidate - 1] -
                        potentialRow[currentRow] - potentialCol[candidate]
                    if (reduced < minValue[candidate]) {
                        minValue[candidate] = reduced
                        path[candidate] = column
                    }
                    if (minValue[candidate] < delta) {
                        delta = minValue[candidate]
                        nextColumn = candidate
                    }
                }
                for (candidate in 0..m) {
                    if (used[candidate]) {
                        potentialRow[columnToRow[candidate]] += delta
                        potentialCol[candidate] -= delta
                    } else {
                        minValue[candidate] -= delta
                    }
                }
                column = nextColumn
            } while (columnToRow[column] != 0)

            do {
                val previous = path[column]
                columnToRow[column] = columnToRow[previous]
                column = previous
            } while (column != 0)
        }
    }

    private fun ensureCapacity(rows: Int, cols: Int) {
        if (costRowCapacity < rows || costColumnCapacity < cols) {
            costRowCapacity = maxOf(costRowCapacity, rows)
            costColumnCapacity = maxOf(costColumnCapacity, cols)
            costs = Array(costRowCapacity) { DoubleArray(costColumnCapacity) }
        }
        if (potentialRow.size < rows + 1) potentialRow = DoubleArray(rows + 1)
        if (potentialCol.size < cols + 1) potentialCol = DoubleArray(cols + 1)
        if (columnToRow.size < cols + 1) columnToRow = IntArray(cols + 1)
        if (path.size < cols + 1) path = IntArray(cols + 1)
        if (minValue.size < cols + 1) minValue = DoubleArray(cols + 1)
        if (used.size < cols + 1) used = BooleanArray(cols + 1)
    }

    companion object {
        /** Returned for a detection that could not be linked with any track. */
        const val UNASSIGNED = -1
    }
}
