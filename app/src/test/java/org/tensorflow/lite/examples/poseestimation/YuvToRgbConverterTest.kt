package org.tensorflow.lite.examples.poseestimation

import android.graphics.Bitmap
import android.graphics.Rect
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Test
import java.nio.ByteBuffer

class YuvToRgbConverterTest {

    @Test
    fun testYuvToRgbConverterWithBitmap() {
        val converter = YuvToRgbConverter()
        val width = 4
        val height = 4

        val yBuf = ByteBuffer.allocateDirect(width * height)
        val uBuf = ByteBuffer.allocateDirect((width / 2) * (height / 2))
        val vBuf = ByteBuffer.allocateDirect((width / 2) * (height / 2))

        for (i in 0 until width * height) {
            yBuf.put(i, 235.toByte()) // pure white luma
        }
        for (i in 0 until (width / 2) * (height / 2)) {
            uBuf.put(i, 128.toByte())
            vBuf.put(i, 128.toByte())
        }

        val planes = arrayOf(
            PlaneData(yBuf, rowStride = width, pixelStride = 1),
            PlaneData(uBuf, rowStride = width / 2, pixelStride = 1),
            PlaneData(vBuf, rowStride = width / 2, pixelStride = 1)
        )

        val output = Bitmap.createBitmap(width, height)
        converter.yuvToRgb(planes, Rect(0, 0, width, height), output, rotationDegrees = 0)

        for (y in 0 until height) {
            for (x in 0 until width) {
                val pixel = output.getPixel(x, y)
                val a = (pixel ushr 24) and 0xFF
                val r = (pixel ushr 16) and 0xFF
                val g = (pixel ushr 8) and 0xFF
                val b = pixel and 0xFF
                assertEquals(255, a)
                assertEquals(255, r)
                assertEquals(255, g)
                assertEquals(255, b)
            }
        }

        converter.release()
    }

    @Test
    fun testYuvToRgbConverterWithRotation() {
        val converter = YuvToRgbConverter()
        val width = 4
        val height = 2

        val yBuf = ByteBuffer.allocateDirect(width * height)
        val uBuf = ByteBuffer.allocateDirect((width / 2) * (height / 2))
        val vBuf = ByteBuffer.allocateDirect((width / 2) * (height / 2))

        // Set one distinctive pixel at (0, 0)
        yBuf.put(0, 16.toByte()) // black at (0,0)
        for (i in 1 until width * height) {
            yBuf.put(i, 235.toByte()) // white elsewhere
        }
        for (i in 0 until (width / 2) * (height / 2)) {
            uBuf.put(i, 128.toByte())
            vBuf.put(i, 128.toByte())
        }

        val planes = arrayOf(
            PlaneData(yBuf, rowStride = width, pixelStride = 1),
            PlaneData(uBuf, rowStride = width / 2, pixelStride = 1),
            PlaneData(vBuf, rowStride = width / 2, pixelStride = 1)
        )

        // 90 degrees rotation: target width = 2, target height = 4
        val output = Bitmap.createBitmap(height, width)
        converter.yuvToRgb(planes, Rect(0, 0, width, height), output, rotationDegrees = 90)

        // Original pixel at (x=0, y=0) rotates to (x' = height - 1 - y = 1, y' = x = 0)
        val blackPixel = output.getPixel(1, 0)
        assertEquals(255, (blackPixel ushr 24) and 0xFF)
        assertEquals(0, (blackPixel ushr 16) and 0xFF)
        assertEquals(0, (blackPixel ushr 8) and 0xFF)
        assertEquals(0, blackPixel and 0xFF)

        // Other pixels should be white
        val whitePixel = output.getPixel(0, 0)
        assertEquals(255, (whitePixel ushr 16) and 0xFF)

        converter.release()
    }
}
