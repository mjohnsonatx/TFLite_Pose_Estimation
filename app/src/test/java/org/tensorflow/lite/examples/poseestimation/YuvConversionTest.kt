package org.tensorflow.lite.examples.poseestimation

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.nio.ByteBuffer
import kotlin.math.abs
import kotlin.math.max

class YuvConversionTest {

    private fun extractA(argb: Int): Int = (argb ushr 24) and 0xFF
    private fun extractR(argb: Int): Int = (argb ushr 16) and 0xFF
    private fun extractG(argb: Int): Int = (argb ushr 8) and 0xFF
    private fun extractB(argb: Int): Int = argb and 0xFF

    @Test
    fun testReferenceValues() {
        // YUV(16, 128, 128) -> RGB(0, 0, 0)
        val black = YuvMath.yuvToRgbPixel(16, 128, 128)
        assertEquals(255, extractA(black))
        assertEquals(0, extractR(black))
        assertEquals(0, extractG(black))
        assertEquals(0, extractB(black))

        // YUV(235, 128, 128) -> RGB(255, 255, 255)
        val white = YuvMath.yuvToRgbPixel(235, 128, 128)
        assertEquals(255, extractA(white))
        assertEquals(255, extractR(white))
        assertEquals(255, extractG(white))
        assertEquals(255, extractB(white))

        // YUV(81, 90, 240) -> approximately RGB(255, 0, 0)
        val red = YuvMath.yuvToRgbPixel(81, 90, 240)
        assertEquals(255, extractA(red))
        assertTrue(abs(extractR(red) - 255) <= 2)
        assertTrue(abs(extractG(red) - 0) <= 2)
        assertTrue(abs(extractB(red) - 0) <= 2)

        // YUV(145, 54, 34) -> approximately RGB(0, 255, 1)
        val green = YuvMath.yuvToRgbPixel(145, 54, 34)
        assertEquals(255, extractA(green))
        assertTrue(abs(extractR(green) - 0) <= 2)
        assertTrue(abs(extractG(green) - 255) <= 2)
        assertTrue(abs(extractB(green) - 1) <= 2)

        // YUV(41, 240, 110) -> approximately RGB(0, 0, 255)
        val blue = YuvMath.yuvToRgbPixel(41, 240, 110)
        assertEquals(255, extractA(blue))
        assertTrue(abs(extractR(blue) - 0) <= 2)
        assertTrue(abs(extractG(blue) - 0) <= 2)
        assertTrue(abs(extractB(blue) - 255) <= 2)
    }

    @Test
    fun testNeutralInputs() {
        for (y in 0..255) {
            val argb = YuvMath.yuvToRgbPixel(y, 128, 128)
            val r = extractR(argb)
            val g = extractG(argb)
            val b = extractB(argb)

            // For neutral inputs where U = V = 128, each channel must match reference and max diff <= 1
            val c = max(0, y - 16)
            val refVal = ((298 * c + 128) shr 8).coerceIn(0, 255)
            assertTrue("R error too large at y=$y", abs(r - refVal) <= 1)
            assertTrue("G error too large at y=$y", abs(g - refVal) <= 1)
            assertTrue("B error too large at y=$y", abs(b - refVal) <= 1)

            val maxDiff = max(abs(r - g), max(abs(g - b), abs(r - b)))
            assertTrue("Channel disparity $maxDiff > 1 at y=$y", maxDiff <= 1)
        }
    }

    @Test
    fun testPlanarAndInterleavedEquivalence() {
        val width = 8
        val height = 6

        // Prepare planar YUV data
        val yData = ByteArray(width * height) { i -> ((i * 17) % 220 + 16).toByte() }
        val uData = ByteArray((width / 2) * (height / 2)) { i -> ((i * 31) % 200 + 20).toByte() }
        val vData = ByteArray((width / 2) * (height / 2)) { i -> ((i * 47) % 200 + 20).toByte() }

        val yBuf = ByteBuffer.allocateDirect(yData.size).apply { put(yData); rewind() }
        val uBuf = ByteBuffer.allocateDirect(uData.size).apply { put(uData); rewind() }
        val vBuf = ByteBuffer.allocateDirect(vData.size).apply { put(vData); rewind() }

        val planarY = PlaneData(yBuf, rowStride = width, pixelStride = 1)
        val planarU = PlaneData(uBuf, rowStride = width / 2, pixelStride = 1)
        val planarV = PlaneData(vBuf, rowStride = width / 2, pixelStride = 1)

        val planarArgb = IntArray(width * height)
        YuvMath.convertYuvToArgb(
            planarY, planarU, planarV,
            cropLeft = 0, cropTop = 0, cropRight = width, cropBottom = height,
            outArgb = planarArgb, rotationDegrees = 0
        )

        // Prepare interleaved UV data (NV12: U then V; NV21: V then U)
        val uvInterleaved = ByteArray((width / 2) * (height / 2) * 2)
        for (i in 0 until (width / 2) * (height / 2)) {
            uvInterleaved[i * 2] = uData[i]
            uvInterleaved[i * 2 + 1] = vData[i]
        }

        val uvBufU = ByteBuffer.allocateDirect(uvInterleaved.size).apply {
            put(uvInterleaved)
            position(0)
        }
        val uvBufV = ByteBuffer.allocateDirect(uvInterleaved.size).apply {
            put(uvInterleaved)
            position(1) // offset by 1 for V
        }

        val interY = PlaneData(yBuf, rowStride = width, pixelStride = 1)
        val interU = PlaneData(uvBufU, rowStride = width, pixelStride = 2, bufferOffset = 0)
        val interV = PlaneData(uvBufV, rowStride = width, pixelStride = 2, bufferOffset = 1)

        val interArgb = IntArray(width * height)
        YuvMath.convertYuvToArgb(
            interY, interU, interV,
            cropLeft = 0, cropTop = 0, cropRight = width, cropBottom = height,
            outArgb = interArgb, rotationDegrees = 0
        )

        for (i in planarArgb.indices) {
            assertEquals("Mismatch at index $i between planar and interleaved", planarArgb[i], interArgb[i])
        }
    }

    @Test
    fun testRowPaddingAndBufferOffsets() {
        val width = 6
        val height = 4
        val yRowStride = 10 // 4 bytes of padding per row
        val uvRowStride = 8 // padding in chroma rows
        val yOffset = 5
        val uOffset = 7
        val vOffset = 3

        val yBuf = ByteBuffer.allocateDirect(yOffset + height * yRowStride)
        val uBuf = ByteBuffer.allocateDirect(uOffset + (height / 2) * uvRowStride)
        val vBuf = ByteBuffer.allocateDirect(vOffset + (height / 2) * uvRowStride)

        // Populate raw data
        for (r in 0 until height) {
            for (c in 0 until width) {
                yBuf.put(yOffset + r * yRowStride + c, (80 + r * 10 + c).toByte())
            }
        }
        for (r in 0 until height / 2) {
            for (c in 0 until width / 2) {
                uBuf.put(uOffset + r * uvRowStride + c, (100 + r * 5 + c).toByte())
                vBuf.put(vOffset + r * uvRowStride + c, (150 + r * 5 + c).toByte())
            }
        }

        val yPlane = PlaneData(yBuf, yRowStride, pixelStride = 1, bufferOffset = yOffset)
        val uPlane = PlaneData(uBuf, uvRowStride, pixelStride = 1, bufferOffset = uOffset)
        val vPlane = PlaneData(vPlane = vBuf, vRowStride = uvRowStride, vPixelStride = 1, vOffset = vOffset)

        val paddedArgb = IntArray(width * height)
        YuvMath.convertYuvToArgb(
            yPlane, uPlane, vPlane,
            cropLeft = 0, cropTop = 0, cropRight = width, cropBottom = height,
            outArgb = paddedArgb, rotationDegrees = 0
        )

        // Verify that every pixel matches direct calculation
        for (r in 0 until height) {
            for (c in 0 until width) {
                val expectedY = 80 + r * 10 + c
                val expectedU = 100 + (r / 2) * 5 + (c / 2)
                val expectedV = 150 + (r / 2) * 5 + (c / 2)
                val expectedArgb = YuvMath.yuvToRgbPixel(expectedY, expectedU, expectedV)
                val actualArgb = paddedArgb[r * width + c]
                assertEquals("Mismatch at ($r, $c)", expectedArgb, actualArgb)
            }
        }
    }

    private fun PlaneData(vPlane: ByteBuffer, vRowStride: Int, vPixelStride: Int, vOffset: Int): PlaneData =
        PlaneData(buffer = vPlane, rowStride = vRowStride, pixelStride = vPixelStride, bufferOffset = vOffset)

    @Test
    fun testCropWithOddOriginAndIrregularDimensions() {
        val srcWidth = 11
        val srcHeight = 9

        val yBuf = ByteBuffer.allocateDirect(srcWidth * srcHeight)
        val uBuf = ByteBuffer.allocateDirect((srcWidth + 1) / 2 * (srcHeight + 1) / 2)
        val vBuf = ByteBuffer.allocateDirect((srcWidth + 1) / 2 * (srcHeight + 1) / 2)

        for (r in 0 until srcHeight) {
            for (c in 0 until srcWidth) {
                yBuf.put(r * srcWidth + c, (50 + r * 7 + c * 3).toByte())
            }
        }
        for (r in 0 until (srcHeight + 1) / 2) {
            for (c in 0 until (srcWidth + 1) / 2) {
                uBuf.put(r * ((srcWidth + 1) / 2) + c, (120 + r * 4 + c * 2).toByte())
                vBuf.put(r * ((srcWidth + 1) / 2) + c, (130 + r * 3 + c * 5).toByte())
            }
        }

        val yPlane = PlaneData(yBuf, srcWidth, 1, 0)
        val uPlane = PlaneData(uBuf, (srcWidth + 1) / 2, 1, 0)
        val vPlane = PlaneData(vBuf, (srcWidth + 1) / 2, 1, 0)

        // Odd crop origin: left = 1, top = 3, right = 6 (width = 5), bottom = 8 (height = 5)
        val cropLeft = 1
        val cropTop = 3
        val cropRight = 6
        val cropBottom = 8
        val cropW = cropRight - cropLeft
        val cropH = cropBottom - cropTop

        val outArgb = IntArray(cropW * cropH)
        YuvMath.convertYuvToArgb(
            yPlane, uPlane, vPlane,
            cropLeft, cropTop, cropRight, cropBottom,
            outArgb, rotationDegrees = 0
        )

        for (y in 0 until cropH) {
            for (x in 0 until cropW) {
                val srcX = cropLeft + x
                val srcY = cropTop + y
                val yVal = 50 + srcY * 7 + srcX * 3
                val uVal = 120 + (srcY / 2) * 4 + (srcX / 2) * 2
                val vVal = 130 + (srcY / 2) * 3 + (srcX / 2) * 5
                val expected = YuvMath.yuvToRgbPixel(yVal, uVal, vVal)
                val actual = outArgb[y * cropW + x]
                assertEquals("Mismatch at cropped ($x, $y)", expected, actual)
            }
        }
    }

    @Test
    fun testRightAngleRotations() {
        val width = 4
        val height = 3

        val yBuf = ByteBuffer.allocateDirect(width * height)
        val uBuf = ByteBuffer.allocateDirect((width / 2) * (height / 2 + 1))
        val vBuf = ByteBuffer.allocateDirect((width / 2) * (height / 2 + 1))

        for (r in 0 until height) {
            for (c in 0 until width) {
                yBuf.put(r * width + c, (16 + r * 20 + c * 10).toByte())
            }
        }
        for (i in 0 until (width / 2) * (height / 2 + 1)) {
            uBuf.put(i, 128.toByte())
            vBuf.put(i, 128.toByte())
        }

        val yPlane = PlaneData(yBuf, width, 1, 0)
        val uPlane = PlaneData(uBuf, width / 2, 1, 0)
        val vPlane = PlaneData(vBuf, width / 2, 1, 0)

        val unrotated = IntArray(width * height)
        YuvMath.convertYuvToArgb(yPlane, uPlane, vPlane, 0, 0, width, height, unrotated, rotationDegrees = 0)

        // 90 degrees clockwise: output width = height = 3, output height = width = 4
        val rot90 = IntArray(width * height)
        YuvMath.convertYuvToArgb(yPlane, uPlane, vPlane, 0, 0, width, height, rot90, rotationDegrees = 90)

        for (y in 0 until height) {
            for (x in 0 until width) {
                val origPixel = unrotated[y * width + x]
                val rotX = height - 1 - y
                val rotY = x
                val rotPixel = rot90[rotY * height + rotX]
                assertEquals("90 deg rotation mapping mismatch at ($x, $y)", origPixel, rotPixel)
            }
        }

        // 180 degrees: output width = 4, height = 3
        val rot180 = IntArray(width * height)
        YuvMath.convertYuvToArgb(yPlane, uPlane, vPlane, 0, 0, width, height, rot180, rotationDegrees = 180)

        for (y in 0 until height) {
            for (x in 0 until width) {
                val origPixel = unrotated[y * width + x]
                val rotX = width - 1 - x
                val rotY = height - 1 - y
                val rotPixel = rot180[rotY * width + rotX]
                assertEquals("180 deg rotation mapping mismatch at ($x, $y)", origPixel, rotPixel)
            }
        }

        // 270 degrees: output width = 3, height = 4
        val rot270 = IntArray(width * height)
        YuvMath.convertYuvToArgb(yPlane, uPlane, vPlane, 0, 0, width, height, rot270, rotationDegrees = 270)

        for (y in 0 until height) {
            for (x in 0 until width) {
                val origPixel = unrotated[y * width + x]
                val rotX = y
                val rotY = width - 1 - x
                val rotPixel = rot270[rotY * height + rotX]
                assertEquals("270 deg rotation mapping mismatch at ($x, $y)", origPixel, rotPixel)
            }
        }
    }
}
