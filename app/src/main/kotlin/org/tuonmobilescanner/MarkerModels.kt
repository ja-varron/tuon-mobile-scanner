package org.tuonmobilescanner

data class DetectedMarker(
    val id: Int,
    val corners: List<NormalizedPoint>
)

data class NormalizedPoint(
    val x: Float,
    val y: Float
)