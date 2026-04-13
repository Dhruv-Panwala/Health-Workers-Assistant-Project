package com.uliza.healthworker.runtime

import android.content.Context
import android.net.Uri
import android.os.ParcelFileDescriptor
import com.arm.aichat.AiChat
import com.arm.aichat.InferenceEngine
import com.arm.aichat.isModelLoaded
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.runBlocking
import java.io.File

data class ModelSupportStatus(
    val sqlModelSupported: Boolean,
    val chatModelSupported: Boolean,
    val sqlModelError: String?,
    val chatModelError: String?,
) {
    val modelsSupported: Boolean
        get() = sqlModelSupported && chatModelSupported

    val summary: String?
        get() {
            val messages = buildList {
                if (!sqlModelSupported && !sqlModelError.isNullOrBlank()) {
                    add("SQL model: ${compactErrorSummary(sqlModelError) ?: "failed"}")
                }
                if (!chatModelSupported && !chatModelError.isNullOrBlank()) {
                    add("Chat model: ${compactErrorSummary(chatModelError) ?: "failed"}")
                }
            }
            return messages.takeIf { it.isNotEmpty() }?.joinToString(" | ")
        }
}

private fun compactErrorSummary(message: String?): String? =
    message
        ?.lineSequence()
        ?.map { it.trim() }
        ?.firstOrNull { it.isNotEmpty() }

private fun describeFailure(error: Throwable): String {
    val messages = mutableListOf<String>()
    var current: Throwable? = error

    while (current != null) {
        val message = current.message
            ?.trim()
            ?.takeIf { it.isNotEmpty() }
            ?: current.javaClass.simpleName

        if (!messages.contains(message)) {
            messages += message
        }
        current = current.cause
    }

    return messages.joinToString("\n\nCaused by:\n")
}

object ModelRuntime {
    private val lock = Any()
    private var engine: InferenceEngine? = null
    private var cachedSupportStatus: ModelSupportStatus? = null

    @Volatile
    private var activeProfile: ModelProfile? = null
    private var loadedProfile: ModelProfile? = null
    private var loadedModelPath: String? = null
    private var loadedSystemPrompt: String? = null

    fun loadedModelName(): String? = loadedProfile?.wireName

    fun generate(
        context: Context,
        runtimePaths: RuntimePaths,
        profile: ModelProfile,
        prompt: String,
        systemPrompt: String,
        maxTokens: Int,
        temperature: Float,
    ): String = synchronized(lock) {
        val inferenceEngine = getEngine(context)
        val modelPath = when (profile) {
            ModelProfile.SQL -> runtimePaths.sqlModelPath
            ModelProfile.CHAT -> runtimePaths.chatModelPath
        }
        val resolvedTemperature = if (temperature > 0f) temperature else profile.defaultTemperature

        activeProfile = profile
        try {
            waitForEngineInitialization(inferenceEngine)
            ensureModelReady(
                context = context,
                inferenceEngine = inferenceEngine,
                profile = profile,
                modelPath = modelPath,
                systemPrompt = systemPrompt,
                temperature = resolvedTemperature,
            )

            val response = StringBuilder()
            runBlocking {
                inferenceEngine
                    .sendUserPrompt(prompt, predictLength = maxTokens.coerceIn(64, 2048))
                    .collect { token -> response.append(token) }
            }
            response.toString().trim()
        } catch (error: Throwable) {
            safeCleanup(inferenceEngine)
            throw error
        } finally {
            activeProfile = null
        }
    }

    fun shutdown() = synchronized(lock) {
        activeProfile = null
        clearLoadedModelState()
        cachedSupportStatus = null
        engine?.destroy()
        engine = null
    }

    fun unloadLoadedModel() = synchronized(lock) {
        activeProfile = null
        engine?.let { inferenceEngine ->
            safeCleanup(inferenceEngine)
        }
    }

    fun modelSupportStatus(context: Context, runtimePaths: RuntimePaths): ModelSupportStatus = synchronized(lock) {
        cachedSupportStatus ?: probeModelSupport(context, runtimePaths).also {
            cachedSupportStatus = it
        }
    }

    private fun getEngine(context: Context): InferenceEngine {
        if (engine == null) {
            engine = AiChat.getInferenceEngine(context.applicationContext)
        }
        return engine!!
    }

    private fun probeModelSupport(context: Context, runtimePaths: RuntimePaths): ModelSupportStatus {
        val sqlProbe = probeSingleModel(
            context = context,
            runtimePaths = runtimePaths,
            profile = ModelProfile.SQL,
            systemPrompt = "You are a SQL model support probe.",
        )
        val chatProbe = probeSingleModel(
            context = context,
            runtimePaths = runtimePaths,
            profile = ModelProfile.CHAT,
            systemPrompt = "You are a chat model support probe.",
        )

        return ModelSupportStatus(
            sqlModelSupported = sqlProbe.first,
            chatModelSupported = chatProbe.first,
            sqlModelError = sqlProbe.second,
            chatModelError = chatProbe.second,
        )
    }

    private fun probeSingleModel(
        context: Context,
        runtimePaths: RuntimePaths,
        profile: ModelProfile,
        systemPrompt: String,
    ): Pair<Boolean, String?> {
        val inferenceEngine = getEngine(context)
        val modelPath = when (profile) {
            ModelProfile.SQL -> runtimePaths.sqlModelPath
            ModelProfile.CHAT -> runtimePaths.chatModelPath
        }

        activeProfile = profile
        return try {
            waitForEngineInitialization(inferenceEngine)
            loadWithFallbacks(
                context = context,
                inferenceEngine = inferenceEngine,
                modelPath = modelPath,
                systemPrompt = systemPrompt,
                temperature = profile.defaultTemperature,
            )
            true to null
        } catch (error: Throwable) {
            safeCleanup(inferenceEngine)
            false to describeFailure(error)
        } finally {
            activeProfile = null
            safeCleanup(inferenceEngine)
        }
    }

    private fun ensureModelReady(
        context: Context,
        inferenceEngine: InferenceEngine,
        profile: ModelProfile,
        modelPath: String,
        systemPrompt: String,
        temperature: Float,
    ) {
        val alreadyLoaded =
            inferenceEngine.state.value.isModelLoaded &&
                loadedProfile == profile &&
                loadedModelPath == modelPath

        if (alreadyLoaded) {
            if (loadedSystemPrompt == systemPrompt) {
                return
            }
            safeCleanup(inferenceEngine)
        }

        loadWithFallbacks(
            context = context,
            inferenceEngine = inferenceEngine,
            modelPath = modelPath,
            systemPrompt = systemPrompt,
            temperature = temperature,
        )
        loadedProfile = profile
        loadedModelPath = modelPath
        loadedSystemPrompt = systemPrompt
    }

    private fun loadWithFallbacks(
        context: Context,
        inferenceEngine: InferenceEngine,
        modelPath: String,
        systemPrompt: String,
        temperature: Float,
    ) {
        val contextFallbacks = listOf(4096, 3072, 2048)
        var lastError: Throwable? = null
        val attemptSummaries = mutableListOf<String>()

        for (contextSize in contextFallbacks) {
            try {
                safeCleanup(inferenceEngine)
                openModelDescriptor(context, modelPath).use { modelDescriptor ->
                    runBlocking {
                        inferenceEngine.loadModel(
                            pathToModel = modelPath,
                            contextSize = contextSize,
                            temperature = temperature,
                            modelFd = modelDescriptor.fd,
                        )
                    }
                }
                runBlocking {
                    inferenceEngine.setSystemPrompt(systemPrompt)
                }
                return
            } catch (error: Throwable) {
                lastError = error
                val attemptMessage = compactErrorSummary(describeFailure(error))
                    ?: error.javaClass.simpleName
                attemptSummaries += "ctx=$contextSize: $attemptMessage"
                safeCleanup(inferenceEngine)

                val loadStageFailure =
                    attemptMessage.startsWith("Failed to load model from") ||
                    attemptMessage.startsWith("Unable to open model descriptor") ||
                    attemptMessage.startsWith("File not found") ||
                    attemptMessage.startsWith("Cannot read file")

                if (loadStageFailure) {
                    break
                }
            }
        }

        val lastFailureMessage = lastError?.let(::describeFailure)
        val usedContextFallbacks = attemptSummaries.size > 1
        val message = buildString {
            append(
                if (usedContextFallbacks) {
                    "Unable to load model after context fallback attempts."
                } else {
                    "Unable to load model."
                }
            )
            if (attemptSummaries.isNotEmpty()) {
                append(' ')
                append(attemptSummaries.joinToString(" | "))
            }
            if (!lastFailureMessage.isNullOrBlank()) {
                append("\n\nLast error:\n")
                append(lastFailureMessage)
            }
        }

        throw IllegalStateException(message, lastError)
    }

    private fun openModelDescriptor(context: Context, modelPath: String): ParcelFileDescriptor {
        if (modelPath.startsWith("content://")) {
            return context.contentResolver.openFileDescriptor(Uri.parse(modelPath), "r")
                ?: throw IllegalStateException("Unable to open model descriptor for $modelPath")
        }

        val modelFile = File(modelPath)
        require(modelFile.exists()) { "Model file not found: $modelPath" }
        return ParcelFileDescriptor.open(modelFile, ParcelFileDescriptor.MODE_READ_ONLY)
    }

    private fun waitForEngineInitialization(inferenceEngine: InferenceEngine) {
        repeat(240) {
            when (val state = inferenceEngine.state.value) {
                is InferenceEngine.State.Initialized,
                is InferenceEngine.State.ModelReady -> return
                is InferenceEngine.State.Error -> throw state.exception
                else -> Thread.sleep(250)
            }
        }
        throw IllegalStateException("Timed out waiting for llama runtime initialization")
    }

    private fun safeCleanup(inferenceEngine: InferenceEngine) {
        var cleanedUp = false
        when (inferenceEngine.state.value) {
            is InferenceEngine.State.Error -> {
                inferenceEngine.cleanUp()
                cleanedUp = true
            }
            else -> {
                if (inferenceEngine.state.value.isModelLoaded) {
                    inferenceEngine.cleanUp()
                    cleanedUp = true
                }
            }
        }

        if (cleanedUp) {
            clearLoadedModelState()
        }
    }

    private fun clearLoadedModelState() {
        loadedProfile = null
        loadedModelPath = null
        loadedSystemPrompt = null
    }
}
