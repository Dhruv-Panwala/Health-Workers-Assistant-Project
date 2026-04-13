package com.uliza.healthworker.runtime

import android.content.Context

object AndroidLlmBridge {
    @Volatile
    private var appContext: Context? = null

    fun initialize(context: Context) {
        appContext = context.applicationContext
    }

    @JvmStatic
    fun generate(
        profileName: String,
        prompt: String,
        systemPrompt: String,
        maxTokens: Int,
        temperature: Float,
    ): String {
        val context = appContext ?: throw IllegalStateException("AndroidLlmBridge is not initialized")
        val runtimePaths = RuntimeAssetManager.resolvePreparedPaths(context)
        return ModelRuntime.generate(
            context = context,
            runtimePaths = runtimePaths,
            profile = ModelProfile.fromWireName(profileName),
            prompt = prompt,
            systemPrompt = systemPrompt,
            maxTokens = maxTokens,
            temperature = temperature,
        )
    }

    @JvmStatic
    fun loadedModel(): String? = ModelRuntime.loadedModelName()

    @JvmStatic
    fun shutdown() {
        ModelRuntime.shutdown()
    }
}
