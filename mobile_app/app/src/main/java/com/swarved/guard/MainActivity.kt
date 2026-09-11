package com.swarved.guard

import android.Manifest
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.graphics.Color
import android.os.Build
import android.os.Bundle
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.localbroadcastmanager.content.LocalBroadcastManager
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.swarved.guard.adapter.IncidentAdapter
import com.swarved.guard.alert.ScamAlertOverlayService
import com.swarved.guard.audio.AudioCaptureService
import com.swarved.guard.model.ScamIncident
import java.util.Locale

class MainActivity : AppCompatActivity() {

    private lateinit var tvRiskPercentage: TextView
    private lateinit var tvRiskStatus: TextView

    // Receives probability updates from AudioCaptureService in real time
    private val inferenceReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            val prob = intent.getFloatExtra(AudioCaptureService.EXTRA_SYNTHETIC_PROBABILITY, 0f)
            updateRiskMeter(prob)
        }
    }

    private val runtimePermissionRequest = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) {
        if (hasPermission(Manifest.permission.RECORD_AUDIO)) {
            requestOverlayPermissionOrStartGuard()
        }
    }

    private val overlayPermissionRequest = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) {
        if (ScamAlertOverlayService.hasOverlayPermission(this)) AudioCaptureService.start(this)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        tvRiskPercentage = findViewById(R.id.tvRiskPercentage)
        tvRiskStatus = findViewById(R.id.tvRiskStatus)

        val recyclerView = findViewById<RecyclerView>(R.id.recyclerIncidents)
        val incidents = listOf(
            ScamIncident("Today • 4:32 PM", "Unknown Caller (Deepfake Impersonation)", 96),
            ScamIncident("Today • 1:18 PM", "+91 98XXXXXX12 (Voice Clone Attack)", 89),
            ScamIncident("Yesterday • 8:45 PM", "Unknown Caller (Digital Arrest Scam)", 94)
        )
        recyclerView.layoutManager = LinearLayoutManager(this)
        recyclerView.adapter = IncidentAdapter(incidents)

        requestRuntimePermissionsOrContinue()
    }

    override fun onResume() {
        super.onResume()
        LocalBroadcastManager.getInstance(this).registerReceiver(
            inferenceReceiver,
            IntentFilter(AudioCaptureService.ACTION_PROBABILITY_UPDATE)
        )
    }

    override fun onPause() {
        super.onPause()
        LocalBroadcastManager.getInstance(this).unregisterReceiver(inferenceReceiver)
    }

    /**
     * Updates the live risk meter card with the latest synthetic-voice probability.
     * Green below 40%, orange 40–70%, red above 70%.
     */
    private fun updateRiskMeter(syntheticProb: Float) {
        val pct = (syntheticProb * 100f).coerceIn(0f, 100f)
        tvRiskPercentage.text = String.format(Locale.US, "%02.0f%%", pct)
        when {
            pct < 40f -> {
                tvRiskPercentage.setTextColor(Color.parseColor("#00E676"))
                tvRiskStatus.setTextColor(Color.parseColor("#00E676"))
                tvRiskStatus.text = "NORMAL HUMAN VOICE"
            }
            pct < 70f -> {
                tvRiskPercentage.setTextColor(Color.parseColor("#FFA726"))
                tvRiskStatus.setTextColor(Color.parseColor("#FFA726"))
                tvRiskStatus.text = "SUSPICIOUS — MONITORING"
            }
            else -> {
                tvRiskPercentage.setTextColor(Color.parseColor("#FF1744"))
                tvRiskStatus.setTextColor(Color.parseColor("#FF1744"))
                tvRiskStatus.text = "⚠ AI SYNTHETIC VOICE DETECTED"
            }
        }
    }

    private fun requestRuntimePermissionsOrContinue() {
        val requiredPermissions = buildList {
            if (!hasPermission(Manifest.permission.RECORD_AUDIO)) add(Manifest.permission.RECORD_AUDIO)
            if (!hasPermission(Manifest.permission.ACCESS_COARSE_LOCATION)) {
                add(Manifest.permission.ACCESS_COARSE_LOCATION)
            }
            // Required so PhoneCallReceiver fires and auto-starts the guard on calls
            if (!hasPermission(Manifest.permission.READ_PHONE_STATE)) {
                add(Manifest.permission.READ_PHONE_STATE)
            }
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU &&
                !hasPermission(Manifest.permission.POST_NOTIFICATIONS)
            ) {
                add(Manifest.permission.POST_NOTIFICATIONS)
            }
        }
        if (requiredPermissions.isNotEmpty()) {
            runtimePermissionRequest.launch(requiredPermissions.toTypedArray())
        } else {
            requestOverlayPermissionOrStartGuard()
        }
    }

    private fun requestOverlayPermissionOrStartGuard() {
        if (ScamAlertOverlayService.hasOverlayPermission(this)) {
            AudioCaptureService.start(this)
        } else {
            overlayPermissionRequest.launch(ScamAlertOverlayService.overlayPermissionIntent(this))
        }
    }

    private fun hasPermission(permission: String): Boolean =
        ContextCompat.checkSelfPermission(this, permission) == PackageManager.PERMISSION_GRANTED
}
