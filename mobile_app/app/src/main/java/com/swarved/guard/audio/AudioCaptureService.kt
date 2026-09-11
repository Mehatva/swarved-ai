package com.swarved.guard.audio

import android.Manifest
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Build
import android.os.IBinder
import android.os.SystemClock
import androidx.core.content.ContextCompat
import androidx.core.app.NotificationCompat
import androidx.localbroadcastmanager.content.LocalBroadcastManager
import com.swarved.guard.R
import com.swarved.guard.alert.ScamAlertOverlayService
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.max

/** Captures microphone input while foreground; private telephone and VoIP audio is not accessible. */
class AudioCaptureService : Service() {
    private val running = AtomicBoolean(false)
    private val captureExecutor: ExecutorService = Executors.newSingleThreadExecutor()
    @Volatile private var audioRecord: AudioRecord? = null
    private var lastAlertAtElapsedMs = -ALERT_COOLDOWN_MS
    private var consecutiveSafeFrames = 0
    private var alertIsArmed = true
    private val probabilityHistory = ArrayDeque<Float>()

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_STOP -> {
                stopCapture()
                stopSelf()
                return START_NOT_STICKY
            }
            ACTION_START, null -> startCaptureIfPossible()
        }
        return START_NOT_STICKY
    }

    override fun onDestroy() {
        stopCapture()
        captureExecutor.shutdownNow()
        NativeVoiceGuard.shutdown()
        super.onDestroy()
    }

    private fun startCaptureIfPossible() {
        if (!running.compareAndSet(false, true)) return
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) !=
            PackageManager.PERMISSION_GRANTED
        ) {
            running.set(false)
            stopSelf()
            return
        }
        startForeground(NOTIFICATION_ID, buildForegroundNotification())
        captureExecutor.execute { initializeAndCapture() }
    }

    /** Runs all potentially slow model, file, and AudioRecord setup work away from the main thread. */
    private fun initializeAndCapture() {
        val modelPath = runCatching { copyModelAssetToPrivateStorage() }.getOrElse {
            stopAfterInitializationFailure()
            return
        }
        if (!running.get() || !NativeVoiceGuard.initialize(modelPath.absolutePath)) {
            stopAfterInitializationFailure()
            return
        }
        val minBufferBytes = AudioRecord.getMinBufferSize(
            SAMPLE_RATE_HZ, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT
        )
        if (minBufferBytes <= 0) {
            stopAfterInitializationFailure()
            return
        }
        val record = AudioRecord.Builder()
            .setAudioSource(MediaRecorder.AudioSource.VOICE_RECOGNITION)
            .setAudioFormat(
                AudioFormat.Builder()
                    .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                    .setSampleRate(SAMPLE_RATE_HZ)
                    .setChannelMask(AudioFormat.CHANNEL_IN_MONO)
                    .build()
            )
            .setBufferSizeInBytes(max(minBufferBytes, FRAME_SAMPLES * PCM_16_BYTES * 2))
            .build()
        if (!running.get() || record.state != AudioRecord.STATE_INITIALIZED) {
            record.release()
            stopAfterInitializationFailure()
            return
        }
        audioRecord = record
        captureLoop(record)
    }

    private fun captureLoop(record: AudioRecord) {
        // CONFIRMED per ML team spec: the model accepts one 48,000-sample (3-second),
        // 16 kHz mono PCM window per inference.
        val pcmFrame = ShortArray(FRAME_SAMPLES)
        var filledSamples = 0
        try {
            record.startRecording()
            while (running.get()) {
                val samplesRead = record.read(
                    pcmFrame,
                    filledSamples,
                    FRAME_SAMPLES - filledSamples,
                    AudioRecord.READ_BLOCKING
                )
                if (samplesRead > 0) {
                    filledSamples += samplesRead
                    if (filledSamples != FRAME_SAMPLES) continue

                    // --- Silence Gate & Peak Normalization ---
                    var maxVal = 0.0f
                    var sumSq = 0.0
                    for (sample in pcmFrame) {
                        val v = sample.toFloat() / 32768f
                        sumSq += (v * v).toDouble()
                        val absV = kotlin.math.abs(v)
                        if (absV > maxVal) maxVal = absV
                    }
                    val rms = kotlin.math.sqrt(sumSq / pcmFrame.size).toFloat()

                    if (rms >= SILENCE_GATE_RMS) {
                        if (maxVal > 0.0001f) {
                            val scale = 0.8f / maxVal
                            for (i in pcmFrame.indices) {
                                var scaled = (pcmFrame[i] * scale).toInt()
                                if (scaled > 32767) scaled = 32767
                                else if (scaled < -32768) scaled = -32768
                                pcmFrame[i] = scaled.toShort()
                            }
                        }

                        NativeVoiceGuard.inferPcm16(pcmFrame)
                            .takeIf { it.isFinite() }
                            ?.let { probability -> handleProbability(probability, pcmFrame) }
                    } else {
                        // When silent, we still need to tell the UI we are alive but detecting no threat.
                        // We pass a 0.0f probability to drag the rolling average down cleanly.
                        handleProbability(0.0f, pcmFrame)
                    }

                    filledSamples = 0
                } else if (samplesRead < 0) {
                    break
                }
            }
        } catch (_: IllegalStateException) {
            // Android may revoke microphone capture while this service is active.
        } finally {
            runCatching { record.stop() }
            record.release()
            if (audioRecord === record) audioRecord = null
            running.set(false)
        }
    }

    private fun handleProbability(probability: Float, triggeringPcm: ShortArray) {
        val now = SystemClock.elapsedRealtime()

        probabilityHistory.addLast(probability)
        if (probabilityHistory.size > SMOOTH_WINDOW) {
            probabilityHistory.removeFirst()
        }
        val smoothedProb = probabilityHistory.average().toFloat()

        // Broadcast every inference result to MainActivity for live risk meter update
        LocalBroadcastManager.getInstance(this).sendBroadcast(
            Intent(ACTION_PROBABILITY_UPDATE)
                .putExtra(EXTRA_SYNTHETIC_PROBABILITY, smoothedProb)
        )

        if (smoothedProb > SYNTHETIC_THRESHOLD) {
            consecutiveSafeFrames = 0
            if (alertIsArmed && now - lastAlertAtElapsedMs >= ALERT_COOLDOWN_MS) {
                lastAlertAtElapsedMs = now
                alertIsArmed = false
                startService(
                    Intent(this, ScamAlertOverlayService::class.java)
                        .setAction(ScamAlertOverlayService.ACTION_SHOW)
                        .putExtra(ScamAlertOverlayService.EXTRA_SYNTHETIC_PROBABILITY, smoothedProb)
                        .putExtra(
                            ScamAlertOverlayService.EXTRA_TRIGGERING_PCM,
                            triggeringPcm.copyOf()
                        )
                )
            }
        } else {
            consecutiveSafeFrames += 1
            if (consecutiveSafeFrames >= SAFE_FRAMES_TO_REARM) alertIsArmed = true
        }
    }

    private fun stopCapture() {
        running.set(false)
        audioRecord?.let { record -> runCatching { record.stop() } }
        audioRecord = null
    }

    private fun stopAfterInitializationFailure() {
        running.set(false)
        stopForeground(STOP_FOREGROUND_REMOVE)
        stopSelf()
    }

    private fun copyModelAssetToPrivateStorage(): File {
        val modelDirectory = File(filesDir, "models")
        if (!modelDirectory.exists() && !modelDirectory.mkdirs()) {
            throw IllegalStateException("Cannot create model directory")
        }
        val destination = File(modelDirectory, MODEL_ASSET_NAME)
        if (destination.exists() && destination.length() > 0L) return destination
        val temporary = File(modelDirectory, "$MODEL_ASSET_NAME.part")
        assets.open(MODEL_ASSET_NAME).use { input ->
            FileOutputStream(temporary).use { output -> input.copyTo(output) }
        }
        if (!temporary.renameTo(destination)) {
            temporary.copyTo(destination, overwrite = true)
            temporary.delete()
        }
        return destination
    }

    private fun buildForegroundNotification(): Notification {
        val manager = getSystemService(NotificationManager::class.java)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            manager.createNotificationChannel(
                NotificationChannel(
                    NOTIFICATION_CHANNEL_ID,
                    getString(R.string.audio_guard_channel_name),
                    NotificationManager.IMPORTANCE_LOW
                )
            )
        }
        return NotificationCompat.Builder(this, NOTIFICATION_CHANNEL_ID)
            .setSmallIcon(R.mipmap.ic_launcher)
            .setContentTitle(getString(R.string.audio_guard_notification_title))
            .setContentText(getString(R.string.audio_guard_notification_text))
            .setOngoing(true)
            .setCategory(NotificationCompat.CATEGORY_SERVICE)
            .build()
    }

    companion object {
        const val ACTION_START = "com.swarved.guard.action.START_AUDIO_GUARD"
        const val ACTION_STOP = "com.swarved.guard.action.STOP_AUDIO_GUARD"
        /** Broadcast action sent to MainActivity after every 3-second inference window. */
        const val ACTION_PROBABILITY_UPDATE = "com.swarved.guard.action.PROBABILITY_UPDATE"
        /** Float extra: synthetic-voice probability in [0, 1]. */
        const val EXTRA_SYNTHETIC_PROBABILITY = "synthetic_probability"
        private const val NOTIFICATION_ID = 101
        private const val NOTIFICATION_CHANNEL_ID = "audio_guard"
        private const val SAMPLE_RATE_HZ = 16_000
        private const val FRAME_SAMPLES = 48_000
        private const val PCM_16_BYTES = 2
        private const val SYNTHETIC_THRESHOLD = 0.85f
        private const val SMOOTH_WINDOW = 5
        private const val SILENCE_GATE_RMS = 0.001f
        // Tunable: 3-second inference windows make this a 9-second safe-audio rearm period.
        private const val SAFE_FRAMES_TO_REARM = 3
        private const val ALERT_COOLDOWN_MS = 30_000L
        private const val MODEL_ASSET_NAME = "swarved_e2e_raw_pcm_int8.onnx"

        fun start(context: Context) {
            ContextCompat.startForegroundService(
                context,
                Intent(context, AudioCaptureService::class.java).setAction(ACTION_START)
            )
        }

        fun stop(context: Context) {
            context.startService(
                Intent(context, AudioCaptureService::class.java).setAction(ACTION_STOP)
            )
        }
    }
}

/** Kotlin boundary for the persistent C++ ONNX Runtime session. */
private object NativeVoiceGuard {
    private val libraryLoaded = runCatching { System.loadLibrary("swarved_voice_guard") }.isSuccess
    @Volatile private var initialized = false

    fun initialize(modelPath: String): Boolean {
        initialized = libraryLoaded && nativeInitialize(modelPath)
        return initialized
    }
    fun inferPcm16(frame: ShortArray): Float = if (initialized) nativeInferPcm16(frame) else Float.NaN
    fun shutdown() {
        if (initialized) nativeShutdown()
        initialized = false
    }

    private external fun nativeInitialize(modelPath: String): Boolean
    private external fun nativeInferPcm16(frame: ShortArray): Float
    private external fun nativeShutdown()
}
