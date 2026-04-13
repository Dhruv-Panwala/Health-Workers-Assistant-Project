package com.uliza.healthworker.plugins

import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import com.getcapacitor.JSArray
import com.getcapacitor.JSObject
import com.getcapacitor.Plugin
import com.getcapacitor.PluginCall
import com.getcapacitor.PluginMethod
import com.getcapacitor.annotation.CapacitorPlugin
import com.uliza.healthworker.runtime.AndroidLlmBridge
import com.uliza.healthworker.runtime.ModelSupportStatus
import com.uliza.healthworker.runtime.ModelRuntime
import com.uliza.healthworker.runtime.RuntimeAssetManager
import java.util.concurrent.Executors
import java.util.Locale

@CapacitorPlugin(name = "UlizaEngine")
class UlizaEnginePlugin : Plugin() {
    private val executor = Executors.newSingleThreadExecutor()
    private val cureMedSystemPrompt =
        "You are CURE-MED, an offline health assistant for community health workers. " +
            "Answer clearly, safely, and practically using general medical knowledge. " +
            "Do not invent patient-specific facts, lab results, or database evidence. " +
            "If you are uncertain, say so plainly and give the safest next step. " +
            "Do not mention prompts, SQL, or models."
    private val openerTokens =
        setOf(
            "hi",
            "hello",
            "hey",
            "habari",
            "jambo",
            "hujambo",
            "mambo",
            "sasa",
            "salama",
            "unaweza",
            "kunisaidia",
            "naweza",
            "msaada",
            "naomba",
            "tafadhali",
            "please",
            "help",
            "can",
            "you",
            "me",
        )

    @PluginMethod
    fun prepareAssets(call: PluginCall) {
        runInBackground(call) {
            AndroidLlmBridge.initialize(context)
            ModelRuntime.shutdown()
            val result = RuntimeAssetManager.prepare(context)
            JSObject().apply {
                put("ready", result.ready)
                put("copiedBytes", result.copiedBytes)
                put("totalBytes", result.totalBytes)
                put("activeDb", result.activeDb)
            }
        }
    }

    @PluginMethod
    fun getBundleInfo(call: PluginCall) {
        runInBackground(call) {
            AndroidLlmBridge.initialize(context)
            val bundleInfo = RuntimeAssetManager.bundleInfo(context)
            JSObject().apply {
                put("totalBytes", bundleInfo.totalBytes)
            }
        }
    }

    @PluginMethod
    fun getRuntimeStatus(call: PluginCall) {
        runInBackground(call) {
            AndroidLlmBridge.initialize(context)
            val status = RuntimeAssetManager.status(context)
            val support =
                if (status.assetsReady) {
                    try {
                        val runtimePaths = RuntimeAssetManager.resolvePreparedPaths(context)
                        ModelRuntime.modelSupportStatus(context, runtimePaths)
                    } catch (error: Throwable) {
                        val message = error.message ?: "Prepared runtime assets could not be resolved."
                        ModelSupportStatus(
                            sqlModelSupported = false,
                            chatModelSupported = false,
                            sqlModelError = message,
                            chatModelError = message,
                        )
                    }
                } else {
                    ModelSupportStatus(
                        sqlModelSupported = false,
                        chatModelSupported = false,
                        sqlModelError = null,
                        chatModelError = null,
                    )
                }
            JSObject().apply {
                put("assetsReady", status.assetsReady)
                put("loadedModel", ModelRuntime.loadedModelName())
                put("modelsSupported", support.modelsSupported)
                put("sqlModelSupported", support.sqlModelSupported)
                put("chatModelSupported", support.chatModelSupported)
                put("sqlModelError", support.sqlModelError)
                put("chatModelError", support.chatModelError)
                put("modelSupportError", support.summary)
            }
        }
    }

    @PluginMethod
    fun query(call: PluginCall) {
        val question = call.getString("question")?.trim().orEmpty()
        val resolvedPlan = call.getString("resolved_plan")?.trim().orEmpty()
        if (question.isBlank() && resolvedPlan.isBlank()) {
            call.reject("Question or resolved plan is required")
            return
        }
        val includeDebugTrace = call.getBoolean("include_debug_trace", false) ?: false
        val requestStartedAt = System.nanoTime()

        runInBackground(call) {
            AndroidLlmBridge.initialize(context)
            val runtimePathStartedAt = System.nanoTime()
            val runtimePaths = RuntimeAssetManager.resolvePreparedPaths(context)
            val runtimePathResolutionMs = elapsedMs(runtimePathStartedAt)

            val pythonBridgeStartedAt = System.nanoTime()
            ensurePythonStarted()

            val python = Python.getInstance()
            val mobileApi = python.getModule("mobile_api")
            mobileApi.callAttr(
                "configure_runtime",
                runtimePaths.activeDbPath,
                runtimePaths.sqlModelPath,
                runtimePaths.chatModelPath,
                true,
            )
            val pythonBridgeEntryMs = elapsedMs(pythonBridgeStartedAt)

            val responseJson = mobileApi.callAttr(
                "answer_question_json",
                question,
                call.getBoolean("debug", false) ?: false,
                call.getInt("page", 1) ?: 1,
                call.getInt("page_size", 100) ?: 100,
                call.getBoolean("include_insights", false) ?: false,
                call.getBoolean("include_rows", true) ?: true,
                call.getBoolean("include_debug_trace", false) ?: false,
                true,
                if (resolvedPlan.isBlank()) null else resolvedPlan,
            ).toString()

            val response = JSObject(responseJson)
            if (includeDebugTrace) {
                attachDebugTimings(
                    response,
                    mapOf(
                        "runtime_path_resolution_ms" to runtimePathResolutionMs,
                        "python_bridge_entry_ms" to pythonBridgeEntryMs,
                        "total_request_ms" to elapsedMs(requestStartedAt),
                    ),
                )
            }

            response
        }
    }

    @PluginMethod
    fun chatConversation(call: PluginCall) {
        val messages = call.getArray("messages")
        if (messages == null || messages.length() == 0) {
            call.reject("Messages are required")
            return
        }

        val language = normalizeLanguage(call.getString("language"))

        runInBackground(call) {
            AndroidLlmBridge.initialize(context)
            val latestUserMessage = latestUserMessage(messages)
            val reply =
                buildOpeningReplyIfApplicable(latestUserMessage, language)
                    ?: run {
                        val prompt = buildConversationPrompt(messages, language)
                        val rawReply = AndroidLlmBridge.generate(
                            profileName = "chat",
                            prompt = prompt,
                            systemPrompt = buildChatSystemPrompt(language),
                            maxTokens = 384,
                            temperature = 0.2f,
                        ).trim()
                        normalizeReply(rawReply, latestUserMessage, language)
                    }

            if (reply.isBlank()) {
                throw IllegalStateException("CURE-MED returned an empty reply.")
            }

            JSObject().apply {
                put("reply", reply)
                put("language", language)
            }
        }
    }

    @PluginMethod
    fun prewarmChatModel(call: PluginCall) {
        runInBackground(call) {
            AndroidLlmBridge.initialize(context)
            val reply = AndroidLlmBridge.generate(
                profileName = "chat",
                prompt = "Reply with the single word READY.",
                systemPrompt = buildChatSystemPrompt("en"),
                maxTokens = 16,
                temperature = 0.2f,
            ).trim()

            JSObject().apply {
                put("ready", reply.isNotBlank())
                put("loadedModel", ModelRuntime.loadedModelName())
            }
        }
    }

    @PluginMethod
    fun unloadModel(call: PluginCall) {
        runInBackground(call) {
            AndroidLlmBridge.initialize(context)
            ModelRuntime.unloadLoadedModel()
            JSObject().apply {
                put("loadedModel", ModelRuntime.loadedModelName())
            }
        }
    }

    override fun handleOnDestroy() {
        super.handleOnDestroy()
        executor.shutdownNow()
        AndroidLlmBridge.shutdown()
    }

    private fun ensurePythonStarted() {
        if (!Python.isStarted()) {
            Python.start(AndroidPlatform(context.applicationContext))
        }
    }

    private fun runInBackground(call: PluginCall, task: () -> JSObject) {
        executor.execute {
            try {
                call.resolve(task())
            } catch (error: Throwable) {
                val exception = if (error is Exception) error else Exception(error)
                call.reject(error.message ?: "UlizaEngine failed", exception)
            }
        }
    }

    private fun normalizeLanguage(language: String?): String =
        when (language?.trim()?.lowercase(Locale.US)) {
            "sw" -> "sw"
            else -> "en"
        }

    private fun buildChatSystemPrompt(language: String): String {
        val languageInstruction =
            if (language == "sw") {
                "Reply only in Kiswahili unless the user explicitly asks to switch languages. " +
                    "Use simple, natural Kiswahili. Keep the reply short and useful. " +
                    "Do not repeat the same phrase, clause, or sentence. " +
                    "If the user is only greeting you or asking whether you can help, greet them once and ask one short follow-up question in Kiswahili."
            } else {
                "Reply only in English unless the user explicitly asks to switch languages. " +
                    "Keep the reply short and useful. " +
                    "Do not repeat the same phrase, clause, or sentence. " +
                    "If the user is only greeting you or asking whether you can help, greet them once and ask one short follow-up question in English."
            }

        return "$cureMedSystemPrompt $languageInstruction"
    }

    private fun buildConversationPrompt(messages: JSArray, language: String): String {
        val lines = mutableListOf<String>()

        for (index in 0 until messages.length()) {
            val message = messages.optJSONObject(index) ?: continue
            val role = message.optString("role").trim().lowercase(Locale.US)
            val content = message.optString("content").trim()

            if (content.isBlank()) {
                continue
            }

            when (role) {
                "user" -> lines += "User: $content"
                "assistant" -> lines += "Assistant: $content"
            }
        }

        require(lines.isNotEmpty()) { "Messages must contain at least one non-empty user or assistant message." }

        return buildString {
            append("Continue this conversation as the assistant.\n")
            append("Write one helpful reply only.\n")
            append(
                if (language == "sw") {
                    "Use Kiswahili only. Keep it natural and avoid repetition.\n\n"
                } else {
                    "Use English only. Keep it natural and avoid repetition.\n\n"
                }
            )
            lines.forEach { line ->
                append(line)
                append("\n\n")
            }
            append("Assistant:")
        }
    }

    private fun latestUserMessage(messages: JSArray): String? {
        for (index in messages.length() - 1 downTo 0) {
            val message = messages.optJSONObject(index) ?: continue
            val role = message.optString("role").trim().lowercase(Locale.US)
            val content = message.optString("content").trim()
            if (role == "user" && content.isNotBlank()) {
                return content
            }
        }
        return null
    }

    private fun buildOpeningReplyIfApplicable(message: String?, language: String): String? {
        if (message.isNullOrBlank() || !isSimpleHelpOpener(message)) {
            return null
        }

        return if (language == "sw") {
            "Ndiyo, naweza kukusaidia. Tafadhali uliza swali lako la afya."
        } else {
            "Yes, I can help. Please ask your health question."
        }
    }

    private fun isSimpleHelpOpener(message: String): Boolean {
        val normalized = message.lowercase(Locale.US)
        val tokens = Regex("[\\p{L}\\p{N}']+").findAll(normalized).map { it.value }.toList()
        if (tokens.isEmpty() || tokens.size > 10) {
            return false
        }

        val tokenSet = tokens.toSet()
        val hasGreeting =
            tokenSet.any { it in setOf("hi", "hello", "hey", "habari", "jambo", "hujambo", "mambo", "sasa", "salama") }
        val hasHelpIntent =
            tokenSet.any { it in setOf("help", "can", "unaweza", "kunisaidia", "msaada") }

        return tokens.all { it in openerTokens } && (hasGreeting || hasHelpIntent)
    }

    private fun normalizeReply(reply: String, latestUserMessage: String?, language: String): String {
        val compact = reply.replace(Regex("\\s+"), " ").trim()
        if (compact.isBlank()) {
            return compact
        }

        if (looksRepetitive(compact)) {
            return buildOpeningReplyIfApplicable(latestUserMessage, language)
                ?: if (language == "sw") {
                    "Ndio, niko tayari kusaidia. Tafadhali eleza swali lako la afya kwa kifupi."
                } else {
                    "I can help. Please describe your health question briefly."
                }
        }

        return compact
    }

    private fun looksRepetitive(reply: String): Boolean {
        val normalized = reply.lowercase(Locale.US)
        if (Regex("(.{12,}?)\\1{2,}").containsMatchIn(normalized)) {
            return true
        }

        val clauses =
            normalized
                .split(Regex("[.!?]+|,"))
                .map { it.trim() }
                .filter { it.length >= 12 }

        val repeatedClauses = clauses.groupingBy { it }.eachCount().values.any { it >= 3 }
        if (repeatedClauses) {
            return true
        }

        val tokens = Regex("[\\p{L}\\p{N}']+").findAll(normalized).map { it.value }.toList()
        if (tokens.size < 8) {
            return false
        }

        val uniqueRatio = tokens.toSet().size.toFloat() / tokens.size.toFloat()
        return uniqueRatio < 0.38f
    }

    private fun elapsedMs(startedAtNanos: Long): Double = (System.nanoTime() - startedAtNanos) / 1_000_000.0

    private fun attachDebugTimings(response: JSObject, timings: Map<String, Double>) {
        val debugTrace = response.optJSONObject("debug_trace")?.let { JSObject(it.toString()) } ?: JSObject()
        val timingObject = debugTrace.optJSONObject("timings")?.let { JSObject(it.toString()) } ?: JSObject()

        timings.forEach { (key, value) ->
            timingObject.put(key, value)
        }

        debugTrace.put("timings", timingObject)
        response.put("debug_trace", debugTrace)
    }
}
