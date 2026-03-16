package com.healthworker.assistant

import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import com.getcapacitor.JSObject
import com.getcapacitor.Plugin
import com.getcapacitor.PluginCall
import com.getcapacitor.PluginMethod
import com.getcapacitor.annotation.CapacitorPlugin

@CapacitorPlugin(name = "OfflineBackend")
open class OfflineBackendPlugin : Plugin() {
    private lateinit var llm: LlmService

    override fun load() {
        super.load()

        if (!Python.isStarted()) {
            Python.start(AndroidPlatform(context))
        }

        DatabaseInstaller.installDatabase(context)
        llm = LlmService(context)
        llm.ensureModelLoaded()
    }

    @PluginMethod
    fun runQuery(call: PluginCall) {
        try {
            val question = call.getString("question") ?: ""
            val debug = call.getBoolean("debug", false) ?: false
            val page = call.getInt("page", 1) ?: 1
            val pageSize = call.getInt("page_size", 200) ?: 200
            val includeInsights = call.getBoolean("include_insights", false) ?: false
            val includeRows = call.getBoolean("include_rows", true) ?: true

            val backend = Python.getInstance().getModule("offline_backend")
            val result = backend.callAttr(
                "run_query",
                question,
                debug,
                page,
                pageSize,
                includeInsights,
                includeRows
            )

            call.resolve(JSObject(result.toString()))
        } catch (e: Exception) {
            call.reject(e.message ?: "Unknown error in runQuery", e)
        }
    }

    @PluginMethod
    fun runLlm(call: PluginCall) {
        val prompt = call.getString("prompt") ?: ""
        val ret = JSObject()
        ret.put("output", llm.infer(prompt))
        call.resolve(ret)
    }

    override fun handleOnDestroy() {
        if (::llm.isInitialized) {
            llm.close()
        }
        super.handleOnDestroy()
    }
}

@CapacitorPlugin(name = "offlineBackend")
class OfflineBackendCompatPlugin : OfflineBackendPlugin()
