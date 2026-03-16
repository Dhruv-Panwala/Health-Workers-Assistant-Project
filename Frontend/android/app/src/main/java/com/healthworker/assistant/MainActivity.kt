package com.healthworker.assistant

import android.os.Bundle
import com.getcapacitor.BridgeActivity

class MainActivity : BridgeActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        registerPlugin(OfflineBackendPlugin::class.java)
        registerPlugin(OfflineBackendCompatPlugin::class.java)
        super.onCreate(savedInstanceState)
    }
}
