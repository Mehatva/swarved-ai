package com.swarved.guard

import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.swarved.guard.alert.ScamAlertOverlayService
import com.swarved.guard.audio.AudioCaptureService
import com.swarved.guard.adapter.IncidentAdapter
import com.swarved.guard.model.ScamIncident

class MainActivity : AppCompatActivity() {
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

    private fun requestRuntimePermissionsOrContinue() {
        val requiredPermissions = buildList {
            if (!hasPermission(Manifest.permission.RECORD_AUDIO)) add(Manifest.permission.RECORD_AUDIO)
            if (!hasPermission(Manifest.permission.ACCESS_COARSE_LOCATION)) {
                add(Manifest.permission.ACCESS_COARSE_LOCATION)
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

@Composable
fun Greeting(name: String, modifier: Modifier = Modifier) {
    Text(
        text = "Hello $name!",
        modifier = modifier
    )
}

@Preview(showBackground = true)
@Composable
fun GreetingPreview() {
    SwarVedGuardTheme {
        Greeting("Android")
    }
}
