package com.swarved.guard.receiver

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.media.AudioManager
import android.telephony.TelephonyManager
import android.util.Log
import com.swarved.guard.audio.AudioCaptureService

/**
 * Listens for cellular call state changes and automatically starts/stops the
 * SwarVed audio guard service when a call becomes active.
 *
 * When an outgoing or incoming call goes off-hook, this receiver:
 *  1. Starts [AudioCaptureService] so inference begins immediately.
 *  2. Attempts to enable the speakerphone so the remote party's audio is
 *     captured by the device microphone (best-effort; falls back gracefully
 *     if the system rejects the request).
 *
 * When the call ends (IDLE state), the service is stopped to release the
 * microphone and conserve battery.
 *
 * Required permissions (declared in AndroidManifest.xml):
 *  - android.permission.READ_PHONE_STATE
 *  - android.permission.RECORD_AUDIO  (already present)
 *  - android.permission.MODIFY_AUDIO_SETTINGS
 */
class PhoneCallReceiver : BroadcastReceiver() {

    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action != TelephonyManager.ACTION_PHONE_STATE_CHANGED) return

        val state = intent.getStringExtra(TelephonyManager.EXTRA_STATE) ?: return

        when (state) {
            TelephonyManager.EXTRA_STATE_OFFHOOK -> onCallActive(context)
            TelephonyManager.EXTRA_STATE_IDLE    -> onCallEnded(context)
            // RINGING state — call not yet accepted, do nothing yet
        }
    }

    // -------------------------------------------------------------------------
    // Private helpers
    // -------------------------------------------------------------------------

    private fun onCallActive(context: Context) {
        Log.i(LOG_TAG, "Call active — starting SwarVed audio guard.")

        // Start the foreground audio capture service (no-op if already running).
        AudioCaptureService.start(context)

        // Try to route audio to the speakerphone so the microphone also picks
        // up the remote party's voice.  This is best-effort: some OEM ROMs
        // reject setSpeakerphoneOn() when MODE_IN_CALL is owned by the
        // telephony stack, so we try MODE_IN_COMMUNICATION first (VoIP mode)
        // and fall back silently if that also fails.
        val audio = context.getSystemService(AudioManager::class.java) ?: run {
            Log.w(LOG_TAG, "AudioManager unavailable — skipping speakerphone enable.")
            return
        }

        try {
            if (!audio.isSpeakerphoneOn) {
                // MODE_IN_COMMUNICATION is the documented mode for apps that
                // perform their own VoIP; it is typically accepted by Android
                // without system-level privileges, unlike MODE_IN_CALL.
                audio.mode = AudioManager.MODE_IN_COMMUNICATION
                audio.isSpeakerphoneOn = true
                Log.i(LOG_TAG, "Speakerphone enabled — remote audio now captured by mic.")
            }
        } catch (e: Exception) {
            // Non-fatal: app still detects audio from the local microphone.
            Log.w(LOG_TAG, "Speakerphone enable failed (${e.message}). " +
                    "User may need to enable speaker manually for remote-voice detection.")
        }
    }

    private fun onCallEnded(context: Context) {
        Log.i(LOG_TAG, "Call ended — stopping SwarVed audio guard.")

        // Restore normal audio mode before stopping the service.
        val audio = context.getSystemService(AudioManager::class.java)
        try {
            audio?.let {
                if (it.isSpeakerphoneOn) it.isSpeakerphoneOn = false
                if (it.mode != AudioManager.MODE_NORMAL) it.mode = AudioManager.MODE_NORMAL
            }
        } catch (e: Exception) {
            Log.w(LOG_TAG, "Audio mode restore failed: ${e.message}")
        }

        AudioCaptureService.stop(context)
    }

    companion object {
        private const val LOG_TAG = "PhoneCallReceiver"
    }
}
