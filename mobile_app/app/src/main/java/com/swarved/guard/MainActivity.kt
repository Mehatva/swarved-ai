package com.swarved.guard

import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
import com.swarved.guard.ui.theme.SwarVedGuardTheme
import com.swarved.guard.alert.ScamAlertOverlayService
import com.swarved.guard.audio.AudioCaptureService

class MainActivity : ComponentActivity() {
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
        enableEdgeToEdge()
        setContent {
            SwarVedGuardTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    Greeting(
                        name = "Android",
                        modifier = Modifier.padding(innerPadding)
                    )
                }
            }
        }
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
