package com.swarved.guard.protection

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.location.LocationManager
import android.provider.Settings
import android.util.Log
import androidx.core.content.ContextCompat
import com.swarved.guard.BuildConfig
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.security.MessageDigest
import java.time.Instant
import java.net.HttpURLConnection
import java.net.URI
import java.net.URL
import java.util.UUID
import kotlin.math.max

/**
 * Prototype-only local protection state. A normal Android app cannot freeze independent payment
 * apps; this supplies SwarVed's 30-minute safety state and reports an incident when configured.
 */
object UPIFreezeManager {
    data class ProtectionState(
        val isActive: Boolean = false,
        val expiresAtEpochMs: Long = 0L,
        val syntheticProbability: Float? = null,
        val reportStatus: ReportStatus = ReportStatus.NOT_SENT
    ) {
        val remainingMillis: Long get() = max(0L, expiresAtEpochMs - System.currentTimeMillis())
        val formattedRemaining: String
            get() {
                val totalSeconds = remainingMillis / 1_000L
                return "%02d:%02d".format(totalSeconds / 60L, totalSeconds % 60L)
            }
    }

    enum class ReportStatus { NOT_SENT, SENDING, SENT, FAILED, NOT_CONFIGURED }

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private val mutableState = MutableStateFlow(ProtectionState())
    val state: StateFlow<ProtectionState> = mutableState.asStateFlow()
    private var expiryJob: Job? = null
    @Volatile private var backendBaseUrl: String = BuildConfig.BACKEND_BASE_URL

    fun configureBackend(baseUrl: String) {
        backendBaseUrl = baseUrl.trim().removeSuffix("/")
    }

    fun activate(context: Context, syntheticProbability: Float, triggeringPcm: ShortArray) {
        val expiresAt = System.currentTimeMillis() + PROTECTION_DURATION_MS
        val reportFrame = triggeringPcm.copyOf()
        mutableState.value = ProtectionState(
            isActive = true,
            expiresAtEpochMs = expiresAt,
            syntheticProbability = syntheticProbability.coerceIn(0f, 1f),
            reportStatus = ReportStatus.SENDING
        )
        scheduleExpiry(expiresAt)
        scope.launch {
            val reportStatus = sendIncident(context.applicationContext, syntheticProbability, reportFrame)
            val current = mutableState.value
            if (current.isActive && current.expiresAtEpochMs == expiresAt) {
                mutableState.value = current.copy(reportStatus = reportStatus)
            }
        }
    }

    fun deactivate() {
        expiryJob?.cancel()
        expiryJob = null
        mutableState.value = ProtectionState()
    }

    private fun scheduleExpiry(expiresAt: Long) {
        expiryJob?.cancel()
        expiryJob = scope.launch {
            while (isActive) {
                val remaining = expiresAt - System.currentTimeMillis()
                if (remaining <= 0L) {
                    if (mutableState.value.expiresAtEpochMs == expiresAt) mutableState.value = ProtectionState()
                    return@launch
                }
                delay(minOf(remaining, 1_000L))
            }
        }
    }

    private suspend fun sendIncident(
        context: Context,
        syntheticProbability: Float,
        triggeringPcm: ShortArray
    ): ReportStatus =
        withContext(Dispatchers.IO) {
            val baseUrl = backendBaseUrl
            if (baseUrl.isBlank()) return@withContext ReportStatus.NOT_CONFIGURED
            val endpoint = runCatching {
                val uri = URI(baseUrl)
                require(uri.scheme == "https" || uri.scheme == "http") { "Unsupported backend URL scheme" }
                URL("${baseUrl.removeSuffix("/")}$DISPATCH_PATH")
            }.getOrElse { return@withContext ReportStatus.FAILED }
            var connection: HttpURLConnection? = null
            try {
                val (latitude, longitude) = lastKnownCoordinatesOrDefault(context)
                val body = JSONObject()
                    .put("incident_id", UUID.randomUUID().toString())
                    .put("timestamp", Instant.now().toString())
                    .put("caller_id", DEMO_CALLER_ID)
                    .put("receiver_device_hash", receiverDeviceHash(context))
                    .put("synthetic_voice_probability", syntheticProbability.coerceIn(0f, 1f))
                    .put("audio_fingerprint_hash", pcmFingerprintHash(triggeringPcm))
                    .put("gps_lat", latitude)
                    .put("gps_lon", longitude)
                    .toString()
                connection = (endpoint.openConnection() as HttpURLConnection).apply {
                    requestMethod = "POST"
                    connectTimeout = CONNECT_TIMEOUT_MS
                    readTimeout = READ_TIMEOUT_MS
                    doOutput = true
                    setRequestProperty("Content-Type", "application/json; charset=utf-8")
                    setFixedLengthStreamingMode(body.toByteArray(Charsets.UTF_8).size)
                }
                connection.outputStream.use { it.write(body.toByteArray(Charsets.UTF_8)) }
                if (connection.responseCode in 200..299) ReportStatus.SENT else ReportStatus.FAILED
            } catch (_: Exception) {
                ReportStatus.FAILED
            } finally {
                connection?.disconnect()
            }
        }

    private fun receiverDeviceHash(context: Context): String {
        val androidId = Settings.Secure.getString(context.contentResolver, Settings.Secure.ANDROID_ID)
        if (androidId.isNullOrBlank()) Log.w(LOG_TAG, "ANDROID_ID unavailable; hashing an empty identifier.")
        return sha256Hex(androidId.orEmpty().toByteArray(Charsets.UTF_8))
    }

    private fun pcmFingerprintHash(pcm: ShortArray): String {
        val digest = MessageDigest.getInstance("SHA-256")
        pcm.forEach { sample ->
            val value = sample.toInt()
            digest.update((value and 0xFF).toByte())
            digest.update(((value ushr 8) and 0xFF).toByte())
        }
        return digest.digest().toHex()
    }

    private fun sha256Hex(value: ByteArray): String =
        MessageDigest.getInstance("SHA-256").digest(value).toHex()

    // MUST VERIFY: 0.0/0.0 is a safe schema fallback, not the intended production location policy.
    private fun lastKnownCoordinatesOrDefault(context: Context): Pair<Double, Double> {
        if (ContextCompat.checkSelfPermission(context, Manifest.permission.ACCESS_COARSE_LOCATION) !=
            PackageManager.PERMISSION_GRANTED
        ) {
            Log.w(LOG_TAG, "Coarse location permission is not granted; reporting 0.0/0.0.")
            return DEFAULT_COORDINATES
        }
        return try {
            val manager = context.getSystemService(Context.LOCATION_SERVICE) as? LocationManager
            if (manager == null) {
                Log.w(LOG_TAG, "LocationManager is unavailable; reporting 0.0/0.0.")
                DEFAULT_COORDINATES
            } else {
                val location = manager.getLastKnownLocation(LocationManager.NETWORK_PROVIDER)
                    ?: manager.getLastKnownLocation(LocationManager.PASSIVE_PROVIDER)
                if (location == null) {
                    Log.w(LOG_TAG, "No last-known location is available; reporting 0.0/0.0.")
                    DEFAULT_COORDINATES
                } else {
                    location.latitude to location.longitude
                }
            }
        } catch (exception: SecurityException) {
            Log.w(LOG_TAG, "Location access was denied; reporting 0.0/0.0.", exception)
            DEFAULT_COORDINATES
        } catch (exception: Exception) {
            Log.w(LOG_TAG, "Last-known location lookup failed; reporting 0.0/0.0.", exception)
            DEFAULT_COORDINATES
        }
    }

    private fun ByteArray.toHex(): String = joinToString("") { byte ->
        "%02x".format(byte.toInt() and 0xFF)
    }

    private const val PROTECTION_DURATION_MS = 30L * 60L * 1_000L
    private const val DISPATCH_PATH = "/api/v1/i4c/dispatch"
    private const val CONNECT_TIMEOUT_MS = 5_000
    private const val READ_TIMEOUT_MS = 5_000
    private const val DEMO_CALLER_ID = "DEMO_SOURCE_ANDROID_PROTOTYPE"
    private const val LOG_TAG = "UPIFreezeManager"
    private val DEFAULT_COORDINATES = 0.0 to 0.0
}
