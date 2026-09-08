package com.swarved.guard.alert

import android.app.Service
import android.content.Context
import android.content.Intent
import android.graphics.Color
import android.graphics.PixelFormat
import android.net.Uri
import android.os.IBinder
import android.provider.Settings
import android.view.Gravity
import android.view.View
import android.view.WindowManager
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView
import com.swarved.guard.protection.UPIFreezeManager
import java.util.Locale

/** A permission-gated, single-instance application overlay for a high-confidence detection. */
class ScamAlertOverlayService : Service() {
    private lateinit var windowManager: WindowManager
    private var overlay: View? = null
    private var statusText: TextView? = null
    private var probabilityText: TextView? = null
    private var probability = 0f
    private var triggeringPcm = ShortArray(0)
    private var countdownUpdater: Runnable? = null

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        windowManager = getSystemService(WINDOW_SERVICE) as WindowManager
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_SHOW) {
            probability = intent.getFloatExtra(EXTRA_SYNTHETIC_PROBABILITY, 0f).coerceIn(0f, 1f)
            triggeringPcm = intent.getShortArrayExtra(EXTRA_TRIGGERING_PCM)?.copyOf() ?: ShortArray(0)
            showOrUpdateOverlay()
        }
        return START_NOT_STICKY
    }

    override fun onDestroy() {
        removeOverlay()
        super.onDestroy()
    }

    private fun showOrUpdateOverlay() {
        if (!Settings.canDrawOverlays(this)) {
            stopSelf()
            return
        }
        if (overlay == null) {
            overlay = buildOverlay().also { view ->
                val params = WindowManager.LayoutParams(
                    WindowManager.LayoutParams.MATCH_PARENT,
                    WindowManager.LayoutParams.WRAP_CONTENT,
                    WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
                    WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN or
                        WindowManager.LayoutParams.FLAG_NOT_TOUCH_MODAL,
                    PixelFormat.TRANSLUCENT
                ).apply { gravity = Gravity.TOP or Gravity.CENTER_HORIZONTAL }
                try {
                    windowManager.addView(view, params)
                } catch (_: RuntimeException) {
                    overlay = null
                    stopSelf()
                }
            }
        }
        probabilityText?.text = String.format(
            Locale.US,
            "Synthetic-voice probability: %.1f%%",
            probability * 100f
        )
        updateProtectionStatus()
    }

    private fun buildOverlay(): View {
        val padding = (20 * resources.displayMetrics.density).toInt()
        return LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(Color.rgb(178, 0, 0))
            setPadding(padding, padding, padding, padding)
            elevation = 24f
            addView(TextView(context).apply {
                text = "CRITICAL ALERT: AI SYNTHETIC VOICE DETECTED! DO NOT TRANSFER MONEY."
                setTextColor(Color.WHITE)
                textSize = 20f
                setTypeface(typeface, android.graphics.Typeface.BOLD)
            })
            probabilityText = TextView(context).apply {
                text = String.format(Locale.US, "Synthetic-voice probability: %.1f%%", probability * 100f)
                setTextColor(Color.WHITE)
                textSize = 14f
                setPadding(0, padding / 2, 0, padding / 2)
            }
            addView(probabilityText)
            statusText = TextView(context).apply {
                setTextColor(Color.WHITE)
                textSize = 16f
            }
            addView(statusText)
            addView(Button(context).apply {
                text = "ACTIVATE 30-MIN UPI PROTECTION"
                setOnClickListener {
                    // ASSUMED: /api/v1/i4c/dispatch accepts UPIFreezeManager's incident JSON body.
                    UPIFreezeManager.activate(applicationContext, probability, triggeringPcm)
                    updateProtectionStatus()
                }
            })
            addView(Button(context).apply {
                text = "DISMISS ALERT"
                setOnClickListener { stopSelf() }
            })
        }
    }

    private fun updateProtectionStatus() {
        val state = UPIFreezeManager.state.value
        statusText?.text = if (state.isActive) {
            "UPI PROTECTION ACTIVE\nTransactions blocked for: ${state.formattedRemaining}"
        } else {
            "UPI protection is not active."
        }
        countdownUpdater?.let { statusText?.removeCallbacks(it) }
        if (state.isActive) {
            countdownUpdater = Runnable { updateProtectionStatus() }.also { updater ->
                statusText?.postDelayed(updater, 1_000L)
            }
        }
    }

    private fun removeOverlay() {
        countdownUpdater?.let { statusText?.removeCallbacks(it) }
        countdownUpdater = null
        overlay?.let { view -> runCatching { windowManager.removeView(view) } }
        overlay = null
        statusText = null
        probabilityText = null
    }

    companion object {
        const val ACTION_SHOW = "com.swarved.guard.action.SHOW_SCAM_ALERT"
        const val EXTRA_SYNTHETIC_PROBABILITY = "synthetic_probability"
        const val EXTRA_TRIGGERING_PCM = "triggering_pcm"

        fun hasOverlayPermission(context: Context): Boolean = Settings.canDrawOverlays(context)

        fun overlayPermissionIntent(context: Context): Intent = Intent(
            Settings.ACTION_MANAGE_OVERLAY_PERMISSION,
            Uri.parse("package:${context.packageName}")
        )
    }
}
