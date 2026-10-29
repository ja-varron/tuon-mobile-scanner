// ===== MarkOverlayView.kt: replace the whole file =====
package org.tuonmobilescanner

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.graphics.RectF
import android.util.AttributeSet
import android.util.TypedValue
import android.view.View

// Viewfinder boxes as fractions (0..1) of the upright analysis frame: left, top, right, bottom.
// One list shared by the detector (what it scans) and the overlay (what you see).
// The bottom row is a guess; adjust all six to your real layout.
val VIEWFINDER_BOXES = listOf(
    RectF(0.00f, 0.00f, 0.22f, 0.12f),   // top-left
    RectF(0.78f, 0.00f, 1.00f, 0.12f),   // top-right
    RectF(0.00f, 0.54f, 0.22f, 0.675f),  // middle-left
    RectF(0.78f, 0.54f, 1.00f, 0.675f),  // middle-right
    RectF(0.00f, 0.88f, 0.22f, 1.00f),   // bottom-left
    RectF(0.78f, 0.88f, 1.00f, 1.00f)    // bottom-right
)

class MarkerOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    private val markerPath = Path()
    private val boxRect = RectF()
    private var markers: List<DetectedMarker> = emptyList()
    private val density = resources.displayMetrics.density

    private val markerPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.GREEN
        style = Paint.Style.STROKE
        strokeWidth = 3f * density
    }

    private val idPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.GREEN
        style = Paint.Style.FILL
        textSize = TypedValue.applyDimension(
            TypedValue.COMPLEX_UNIT_SP, 18f, resources.displayMetrics
        )
    }

    // Translucent white fill so the paper stays visible through the box.
    private val boxFillPaint = Paint().apply {
        color = Color.WHITE
        alpha = 70
        style = Paint.Style.FILL
    }

    private val boxStrokePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        style = Paint.Style.STROKE
        strokeWidth = 2f * density
    }

    // Border turns green while a marker sits inside the box.
    private val boxHitPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.GREEN
        style = Paint.Style.STROKE
        strokeWidth = 4f * density
    }

    // 3:4 portrait until the first analyzed frame reports the real size,
    // so the boxes are visible before any detection happens.
    private var frameWidth = 3
    private var frameHeight = 4

    fun updateMarkers(markers: List<DetectedMarker>, frameWidth: Int, frameHeight: Int) {
        this.markers = markers
        this.frameWidth = frameWidth.coerceAtLeast(1)
        this.frameHeight = frameHeight.coerceAtLeast(1)
        invalidate()
    }

    // Marker counts as inside when its center falls in the box.
    private fun isInside(marker: DetectedMarker, box: RectF): Boolean {
        if (marker.corners.size < 4) return false
        val cx = marker.corners.sumOf { it.x.toDouble() }.toFloat() / marker.corners.size
        val cy = marker.corners.sumOf { it.y.toDouble() }.toFloat() / marker.corners.size
        return box.contains(cx, cy)
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        val scale = minOf(width.toFloat() / frameWidth, height.toFloat() / frameHeight)
        val offsetX = (width - frameWidth * scale) / 2f
        val offsetY = (height - frameHeight * scale) / 2f

        // White viewfinder boxes first, so marker outlines draw on top.
        VIEWFINDER_BOXES.forEach { box ->
            boxRect.set(
                offsetX + box.left * frameWidth * scale,
                offsetY + box.top * frameHeight * scale,
                offsetX + box.right * frameWidth * scale,
                offsetY + box.bottom * frameHeight * scale
            )
            val hit = markers.any { isInside(it, box) }
            canvas.drawRect(boxRect, boxFillPaint)
            canvas.drawRect(boxRect, if (hit) boxHitPaint else boxStrokePaint)
        }

        markers.forEach { marker ->
            val corners = marker.corners
            if (corners.size < 4) return@forEach

            val firstPoint = corners[0]

            markerPath.reset()
            markerPath.moveTo(
                offsetX + firstPoint.x * frameWidth * scale,
                offsetY + firstPoint.y * frameHeight * scale
            )
            for (index in 1 until 4) {
                val point = corners[index]
                markerPath.lineTo(
                    offsetX + point.x * frameWidth * scale,
                    offsetY + point.y * frameHeight * scale
                )
            }
            markerPath.close()
            canvas.drawPath(markerPath, markerPaint)

            canvas.drawText(
                marker.id.toString(),
                offsetX + firstPoint.x * frameWidth * scale,
                offsetY + firstPoint.y * frameHeight * scale,
                idPaint
            )
        }
    }
}