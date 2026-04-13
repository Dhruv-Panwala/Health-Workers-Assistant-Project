package com.uliza.healthworker;

import android.os.Bundle;

import com.getcapacitor.BridgeActivity;
import com.uliza.healthworker.plugins.UlizaEnginePlugin;

public class MainActivity extends BridgeActivity {
    @Override
    public void onCreate(Bundle savedInstanceState) {
        registerPlugin(UlizaEnginePlugin.class);
        super.onCreate(savedInstanceState);
    }
}
