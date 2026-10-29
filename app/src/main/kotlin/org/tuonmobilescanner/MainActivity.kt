// app/src/main/kotlin/org/tuonmobilescanner/MainActivity.kt
package org.tuonmobilescanner

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Size
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.annotation.OptIn
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.FocusMeteringAction
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import org.opencv.android.OpenCVLoader
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Rect
import org.opencv.imgproc.Imgproc
import org.opencv.objdetect.ArucoDetector
import org.opencv.objdetect.Dictionary
import org.opencv.objdetect.Objdetect
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var markerOverlay: MarkerOverlayView
    private lateinit var statusText: TextView
    private lateinit var cameraExecutor: ExecutorService

    private val cameraPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) {
                startCamera()
            } else {
                statusText.text = buildString {
                    append("Camera permission is required")
                }
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        previewView = findViewById(R.id.previewView)
        markerOverlay = findViewById(R.id.markerOverlay)
        statusText = findViewById(R.id.statusText)
        cameraExecutor = Executors.newSingleThreadExecutor()

        if (!OpenCVLoader.initLocal()) {
            statusText.text = buildString {
                append("Unable to initialize scanner")
            }
            return
        }

        if (ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.CAMERA
            ) == PackageManager.PERMISSION_GRANTED
        ) {
            startCamera()
        } else {
            cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    // Binds a 4:3 preview and analysis stream with the same aspect ratio so the
    // overlay lines up with what the detector sees.
    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener(
            {
                val cameraProvider = cameraProviderFuture.get()

                // Must match MarkerOverlayView's letterbox math (min scale, centered).
                previewView.scaleType = PreviewView.ScaleType.FIT_CENTER

                val ratio4x3 = AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY

                val preview = Preview.Builder()
                    .setResolutionSelector(
                        ResolutionSelector.Builder()
                            .setAspectRatioStrategy(ratio4x3)
                            .build()
                    )
                    .build()
                    .also { it.surfaceProvider = previewView.surfaceProvider }

                val analysis = ImageAnalysis.Builder()
                    .setResolutionSelector(
                        ResolutionSelector.Builder()
                            .setAspectRatioStrategy(ratio4x3)
                            .setResolutionStrategy(
                                ResolutionStrategy(
                                    Size(1920, 1440),
                                    ResolutionStrategy.FALLBACK_RULE_CLOSEST_LOWER
                                )
                            )
                            .build()
                    )
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                    .build()

                analysis.setAnalyzer(
                    cameraExecutor,
                    ArucoAnalyzer { result ->
                        runOnUiThread {
                            markerOverlay.updateMarkers(
                                result.markers,
                                result.frameWidth,
                                result.frameHeight
                            )

                            statusText.text = when {
                                result.allMarkersVisible -> "All six markers detected"
                                result.detectedIds.isEmpty() -> "Align all six markers"
                                else -> "${result.detectedIds.size}/6 markers detected"
                            }
                        }
                    }
                )

                cameraProvider.unbindAll()

                val camera = cameraProvider.bindToLifecycle(
                    this,
                    CameraSelector.DEFAULT_BACK_CAMERA,
                    preview,
                    analysis
                )

                // post{} waits for layout so width/height are not 0.
                previewView.post {
                    val center = previewView.meteringPointFactory.createPoint(
                        previewView.width / 2f,
                        previewView.height / 2f
                    )
                    camera.cameraControl.startFocusAndMetering(
                        FocusMeteringAction.Builder(
                            center,
                            FocusMeteringAction.FLAG_AF or FocusMeteringAction.FLAG_AE
                        ).build()
                    )
                }
            },
            ContextCompat.getMainExecutor(this)
        )
    }

    override fun onDestroy() {
        cameraExecutor.shutdown()
        super.onDestroy()
    }
}

// Copies only the Y plane from CameraX into a compact grayscale matrix.
@OptIn(ExperimentalGetImage::class)
private fun ImageProxy.toGrayMat(): Mat? {
    val image = this.image ?: return null
    val plane = image.planes[0]
    val w = image.width
    val h = image.height
    val yBuffer = plane.buffer.duplicate().also { it.rewind() }
    val gray = Mat(h, w, CvType.CV_8UC1)

    // Fast path: one bulk copy instead of ~1440 JNI calls.
    if (plane.pixelStride == 1 && plane.rowStride == w) {
        val bytes = ByteArray(w * h)
        yBuffer.get(bytes)
        gray.put(0, 0, bytes)
        return gray
    }

    // Padded rows or unusual pixel stride: copy row by row.
    val row = ByteArray(w)
    for (y in 0 until h) {
        yBuffer.position(y * plane.rowStride)
        if (plane.pixelStride == 1) {
            yBuffer.get(row, 0, w)
        } else {
            for (x in 0 until w) {
                row[x] = yBuffer.get(y * plane.rowStride + x * plane.pixelStride)
            }
        }
        gray.put(y, 0, row)
    }
    return gray
}

// One analyzer result. allMarkersVisible is decided by the analyzer
// (held markers plus a consecutive-frame streak).
private data class DetectionResult(
    val markers: List<DetectedMarker>,
    val detectedIds: Set<Int>,
    val frameWidth: Int,
    val frameHeight: Int,
    val allMarkersVisible: Boolean
)

// Detects markers only inside the viewfinder boxes and keeps the best corners per ID.
private class ArucoAnalyzer(
    private val onResult: (DetectionResult) -> Unit
) : ImageAnalysis.Analyzer {

    // <<< COMPANION OBJECT: goes at the top of ArucoAnalyzer, before the fields.
    // The box list is VIEWFINDER_BOXES in MarkOverlayView.kt, not here.
    companion object {
        val EXPECTED_IDS = setOf(0, 1, 2, 3, 4, 5)
        const val HOLD_FRAMES = 5      // frames a marker stays "seen" after a miss
        const val READY_STREAK = 3     // consecutive complete frames before "ready"
        const val TILE_SCALE = 2.0     // enlargement of each viewfinder crop
    }

    private val dictionary: Dictionary =
        Objdetect.getPredefinedDictionary(Objdetect.DICT_4X4_50)
    private val detector = ArucoDetector(dictionary, detectorParameters())
    private val previousMarkers = mutableMapOf<Int, DetectedMarker>()
    private var frameIndex = 0L
    private var fullStreak = 0
    private val lastSeen = mutableMapOf<Int, Long>()

    private fun detectorParameters() =
        org.opencv.objdetect.DetectorParameters().apply {
            _adaptiveThreshWinSizeMin = 3
            _adaptiveThreshWinSizeMax = 53
            _adaptiveThreshWinSizeStep = 10
            _minMarkerPerimeterRate = 0.02
            _maxMarkerPerimeterRate = 4.0
            _minDistanceToBorder = 1
            _minCornerDistanceRate = 0.02
            _errorCorrectionRate = 0.4
            _cornerRefinementMethod = Objdetect.CORNER_REFINE_SUBPIX
            _cornerRefinementWinSize = 5
            _cornerRefinementMaxIterations = 30
            _cornerRefinementMinAccuracy = 0.01
        }

    override fun analyze(imageProxy: ImageProxy) {
        val gray = imageProxy.toGrayMat()
        if (gray == null) {
            imageProxy.close()
            return
        }

        try {
            when (imageProxy.imageInfo.rotationDegrees) {
                90 -> Core.rotate(gray, gray, Core.ROTATE_90_CLOCKWISE)
                180 -> Core.rotate(gray, gray, Core.ROTATE_180)
                270 -> Core.rotate(gray, gray, Core.ROTATE_90_COUNTERCLOCKWISE)
            }

            val width = gray.cols().toFloat()
            val height = gray.rows().toFloat()
            val candidates = linkedMapOf<Int, DetectionCandidate>()

            val views = createViewfinderViews(gray)
            try {
                views.forEach { detectCandidates(it, width, height, candidates) }
            } finally {
                views.forEach { it.image.release() }
            }

            frameIndex++
            candidates.keys.forEach { lastSeen[it] = frameIndex }
            lastSeen.entries.removeAll { frameIndex - it.value > HOLD_FRAMES }
            previousMarkers.keys.retainAll(lastSeen.keys)

            val visibleIds = lastSeen.keys.toSet()
            fullStreak = if (visibleIds.containsAll(EXPECTED_IDS)) fullStreak + 1 else 0

            val markers = visibleIds.sorted().mapNotNull { id ->
                candidates[id]?.let { stabilizeMarker(it.marker) } ?: previousMarkers[id]
            }

            onResult(
                DetectionResult(
                    markers = markers,
                    detectedIds = visibleIds,
                    frameWidth = gray.cols(),
                    frameHeight = gray.rows(),
                    allMarkersVisible = fullStreak >= READY_STREAK
                )
            )
        } finally {
            gray.release()
            imageProxy.close()
        }
    }

    // <<< createViewfinderViews: goes inside ArucoAnalyzer, right after analyze().
    // Crops each light box, normalizes contrast and enlarges it. Anything outside
    // the boxes is never passed to the detector.
    private fun createViewfinderViews(gray: Mat): List<DetectionView> {
        val clahe = Imgproc.createCLAHE(2.0, org.opencv.core.Size(4.0, 4.0))
        try {
            return VIEWFINDER_BOXES.map { box ->
                val x = (box.left * gray.cols()).toInt().coerceIn(0, gray.cols() - 1)
                val y = (box.top * gray.rows()).toInt().coerceIn(0, gray.rows() - 1)
                val w = ((box.right - box.left) * gray.cols()).toInt().coerceIn(1, gray.cols() - x)
                val h = ((box.bottom - box.top) * gray.rows()).toInt().coerceIn(1, gray.rows() - y)

                val roi = Mat(gray, Rect(x, y, w, h))
                val normalized = Mat()
                clahe.apply(roi, normalized)
                val enlarged = Mat()
                // Empty Size() makes fx/fy take effect.
                Imgproc.resize(
                    normalized,
                    enlarged,
                    org.opencv.core.Size(),
                    TILE_SCALE,
                    TILE_SCALE,
                    Imgproc.INTER_LINEAR
                )
                roi.release()
                normalized.release()
                DetectionView(enlarged, TILE_SCALE, x, y)
            }
        } finally {
            clahe.collectGarbage()
        }
    }

    // Blends each detection with its previous position to remove corner jitter.
    private fun stabilizeMarker(marker: DetectedMarker): DetectedMarker {
        val previous = previousMarkers[marker.id]
        if (previous == null) {
            previousMarkers[marker.id] = marker
            return marker
        }

        val smoothedCorners = marker.corners.mapIndexed { index, current ->
            val old = previous.corners[index]
            val movement = kotlin.math.hypot(
                (current.x - old.x).toDouble(),
                (current.y - old.y).toDouble()
            )
            val alpha = when {
                movement > 0.12 -> 0.65f
                movement > 0.04 -> 0.4f
                else -> 0.22f
            }

            NormalizedPoint(
                x = old.x + (current.x - old.x) * alpha,
                y = old.y + (current.y - old.y) * alpha
            )
        }
        return DetectedMarker(marker.id, smoothedCorners).also {
            previousMarkers[marker.id] = it
        }
    }

    // Runs one detector pass and keeps the best candidate per expected ID.
    private fun detectCandidates(
        detectionView: DetectionView,
        width: Float,
        height: Float,
        candidates: MutableMap<Int, DetectionCandidate>
    ) {
        val passCorners = ArrayList<Mat>()
        val passIds = Mat()

        try {
            detector.detectMarkers(detectionView.image, passCorners, passIds)
            for (index in passCorners.indices) {
                if (index >= passIds.rows()) break

                val id = passIds.get(index, 0)[0].toInt()
                if (id !in EXPECTED_IDS) continue

                val normalizedCorners = normalizeCorners(
                    markerCorners = passCorners[index],
                    scale = detectionView.scale,
                    offsetX = detectionView.offsetX,
                    offsetY = detectionView.offsetY,
                    width = width,
                    height = height
                ) ?: continue

                val marker = DetectedMarker(id, normalizedCorners)
                val score = cornerQuality(normalizedCorners)
                val previous = candidates[id]
                val previousMarker = previousMarkers[id]
                val continuity = previousMarker?.let {
                    cornerDistance(normalizedCorners, it.corners)
                } ?: 0.0
                val candidateScore = score - continuity * 2.0
                if (previous == null || candidateScore > previous.score) {
                    candidates[id] = DetectionCandidate(marker, candidateScore)
                }
            }
        } finally {
            passCorners.forEach(Mat::release)
            passIds.release()
        }
    }

    private fun cornerDistance(
        current: List<NormalizedPoint>,
        previous: List<NormalizedPoint>
    ): Double {
        return current.indices
            .map { index ->
                kotlin.math.hypot(
                    (current[index].x - previous[index].x).toDouble(),
                    (current[index].y - previous[index].y).toDouble()
                )
            }
            .average()
    }

    // Converts crop coordinates back to full-frame coordinates, normalized to 0..1.
    private fun normalizeCorners(
        markerCorners: Mat,
        scale: Double,
        offsetX: Int,
        offsetY: Int,
        width: Float,
        height: Float
    ): List<NormalizedPoint>? {
        return (0 until 4).map { cornerIndex ->
            val point = markerCorners.get(0, cornerIndex) ?: return null
            NormalizedPoint(
                x = ((point[0] / scale + offsetX) / width).coerceIn(0.0, 1.0).toFloat(),
                y = ((point[1] / scale + offsetY) / height).coerceIn(0.0, 1.0).toFloat()
            )
        }
    }

    private fun cornerQuality(corners: List<NormalizedPoint>): Double {
        val sideLengths = corners.indices.map { index ->
            val current = corners[index]
            val next = corners[(index + 1) % corners.size]
            val dx = next.x - current.x
            val dy = next.y - current.y
            kotlin.math.sqrt((dx * dx + dy * dy).toDouble())
        }
        val mean = sideLengths.average()
        if (mean == 0.0) return 0.0
        val variation = sideLengths.sumOf { kotlin.math.abs(it - mean) } / mean
        return mean - variation
    }

    private data class DetectionCandidate(
        val marker: DetectedMarker,
        val score: Double
    )

    // One detector image plus the scale/offset that maps it back to the camera frame.
    private data class DetectionView(
        val image: Mat,
        val scale: Double,
        val offsetX: Int,
        val offsetY: Int
    )
}