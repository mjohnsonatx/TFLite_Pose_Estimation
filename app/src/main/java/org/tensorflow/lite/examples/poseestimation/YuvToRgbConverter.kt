package org.tensorflow.lite.examples.poseestimation

import android.content.Context
import android.graphics.Bitmap
import android.graphics.ImageFormat
import android.graphics.Rect
import android.media.Image
import java.nio.ByteBuffer

/**
 * Encapsulates the buffer and strides of a single image plane for YUV conversion.
 */
data class PlaneData(
    val buffer: ByteBuffer,
    val rowStride: Int,
    val pixelStride: Int,
    val bufferOffset: Int = buffer.position()
)

/**
 * Pure math and pixel transformation utilities for limited-range BT.601 YUV_420_888 to RGB conversion.
 */
object YuvMath {

    /**
     * Converts a single Y, U, V unsigned byte tuple (0..255) to an opaque ARGB-8888 Int.
     * Uses limited-range BT.601 integer math.
     */
    fun yuvToRgbPixel(y: Int, u: Int, v: Int): Int {
        val c = if (y - 16 > 0) y - 16 else 0
        val d = u - 128
        val e = v - 128

        var r = (298 * c + 409 * e + 128) shr 8
        var g = (298 * c - 100 * d - 208 * e + 128) shr 8
        var b = (298 * c + 516 * d + 128) shr 8

        if (r < 0) r = 0 else if (r > 255) r = 255
        if (g < 0) g = 0 else if (g > 255) g = 255
        if (b < 0) b = 0 else if (b > 255) b = 255

        return (0xFF shl 24) or (r shl 16) or (g shl 8) or b
    }

    /**
     * Converts YUV planes with crop and optional right-angle rotation into an ARGB IntArray.
     */
    fun convertYuvToArgb(
        yPlane: PlaneData,
        uPlane: PlaneData,
        vPlane: PlaneData,
        cropLeft: Int,
        cropTop: Int,
        cropRight: Int,
        cropBottom: Int,
        outArgb: IntArray,
        rotationDegrees: Int = 0
    ) {
        val cropWidth = cropRight - cropLeft
        val cropHeight = cropBottom - cropTop
        if (cropWidth <= 0 || cropHeight <= 0) return

        val normalizedRotation = ((rotationDegrees % 360) + 360) % 360
        val outWidth = if (normalizedRotation == 90 || normalizedRotation == 270) cropHeight else cropWidth
        val outHeight = if (normalizedRotation == 90 || normalizedRotation == 270) cropWidth else cropHeight

        require(outArgb.size >= outWidth * outHeight) {
            "outArgb buffer too small: required ${outWidth * outHeight}, got ${outArgb.size}"
        }

        val yBuf = yPlane.buffer
        val uBuf = uPlane.buffer
        val vBuf = vPlane.buffer

        val yRowStride = yPlane.rowStride
        val yPixelStride = yPlane.pixelStride
        val yOffset = yPlane.bufferOffset

        val uRowStride = uPlane.rowStride
        val uPixelStride = uPlane.pixelStride
        val uOffset = uPlane.bufferOffset

        val vRowStride = vPlane.rowStride
        val vPixelStride = vPlane.pixelStride
        val vOffset = vPlane.bufferOffset

        for (y in 0 until cropHeight) {
            val srcY = cropTop + y
            val chromaY = srcY shr 1

            val yRowStart = yOffset + srcY * yRowStride
            val uRowStart = uOffset + chromaY * uRowStride
            val vRowStart = vOffset + chromaY * vRowStride

            for (x in 0 until cropWidth) {
                val srcX = cropLeft + x
                val chromaX = srcX shr 1

                val yIdx = yRowStart + srcX * yPixelStride
                val uIdx = uRowStart + chromaX * uPixelStride
                val vIdx = vRowStart + chromaX * vPixelStride

                val yVal = yBuf.get(yIdx).toInt() and 0xFF
                val uVal = uBuf.get(uIdx).toInt() and 0xFF
                val vVal = vBuf.get(vIdx).toInt() and 0xFF

                val argb = yuvToRgbPixel(yVal, uVal, vVal)

                val destIndex = when (normalizedRotation) {
                    90 -> x * outWidth + (cropHeight - 1 - y)
                    180 -> (cropHeight - 1 - y) * outWidth + (cropWidth - 1 - x)
                    270 -> (cropWidth - 1 - x) * outWidth + y
                    else -> y * outWidth + x
                }

                outArgb[destIndex] = argb
            }
        }
    }
}

/**
 * Reusable YUV to RGB converter compatible with API 23+ that avoids allocations in frame loops.
 */
class YuvToRgbConverter(context: Context? = null) {

    private var argbBuffer: IntArray? = null
    private var bufferWidth: Int = 0
    private var bufferHeight: Int = 0

    @Synchronized
    fun yuvToRgb(image: Image, output: Bitmap, rotationDegrees: Int = 0) {
        val crop = image.cropRect
        val planes = image.planes
        val planeData = Array(planes.size) { i ->
            PlaneData(
                buffer = planes[i].buffer,
                rowStride = planes[i].rowStride,
                pixelStride = planes[i].pixelStride,
                bufferOffset = planes[i].buffer.position()
            )
        }
        yuvToRgb(planeData, crop, output, rotationDegrees)
    }

    @Synchronized
    fun yuvToRgb(
        planes: Array<PlaneData>,
        cropRect: Rect,
        output: Bitmap,
        rotationDegrees: Int = 0
    ) {
        require(planes.size >= 3) { "Image must have at least 3 planes" }
        val cropWidth = cropRect.width()
        val cropHeight = cropRect.height()
        val normalizedRotation = ((rotationDegrees % 360) + 360) % 360
        val outWidth = if (normalizedRotation == 90 || normalizedRotation == 270) cropHeight else cropWidth
        val outHeight = if (normalizedRotation == 90 || normalizedRotation == 270) cropWidth else cropHeight

        val requiredSize = outWidth * outHeight
        if (argbBuffer == null || argbBuffer!!.size < requiredSize || bufferWidth != outWidth || bufferHeight != outHeight) {
            argbBuffer = IntArray(requiredSize)
            bufferWidth = outWidth
            bufferHeight = outHeight
        }

        val buffer = argbBuffer!!
        YuvMath.convertYuvToArgb(
            yPlane = planes[0],
            uPlane = planes[1],
            vPlane = planes[2],
            cropLeft = cropRect.left,
            cropTop = cropRect.top,
            cropRight = cropRect.right,
            cropBottom = cropRect.bottom,
            outArgb = buffer,
            rotationDegrees = normalizedRotation
        )

        output.setPixels(buffer, 0, outWidth, 0, 0, outWidth, outHeight)
    }

    @Synchronized
    fun release() {
        argbBuffer = null
        bufferWidth = 0
        bufferHeight = 0
    }
}
