package com.healthworker.assistant

import android.content.Context
import android.util.Log
import java.io.File
import java.io.FileNotFoundException
import java.io.FileOutputStream

class LlmService(private val context: Context) {

    companion object {
        private const val TAG = "LlmService"
        private const val MODEL_FILE_NAME = "llama-3-sqlcoder-8b-Q4_K_M.gguf"
        private val BUNDLED_MODEL_ASSET_PATHS = arrayOf("models/$MODEL_FILE_NAME", MODEL_FILE_NAME)
        private var nativeLoaded = false

        fun ensureNativeLoaded(): Boolean {
            if (nativeLoaded) return true
            return try {
                System.loadLibrary("llama")
                System.loadLibrary("llama_jni")
                nativeLoaded = true
                true
            } catch (e: Throwable) {
                Log.e(TAG, "Native libs not loaded", e)
                false
            }
        }
    }

    init {
        ensureNativeLoaded()
    }

    private var modelPtr: Long = 0L

    external fun initModel(path: String): Long
    external fun inferNative(modelPtr: Long, prompt: String): String
    external fun releaseModel(modelPtr: Long)

    private fun bundledModelTarget(): File = File(context.filesDir, MODEL_FILE_NAME)

    private fun externalModelCandidate(): File? =
        context.getExternalFilesDir(null)?.let { File(it, "models/$MODEL_FILE_NAME") }

    private fun copyBundledModelIfAvailable(outFile: File): Boolean {
        outFile.parentFile?.mkdirs()

        for (assetPath in BUNDLED_MODEL_ASSET_PATHS) {
            try {
                context.assets.open(assetPath).use { input ->
                    FileOutputStream(outFile).use { output ->
                        input.copyTo(output)
                    }
                }
                return true
            } catch (_: FileNotFoundException) {
                // Keep looking for the model in alternate asset locations.
            } catch (e: Exception) {
                Log.w(TAG, "Failed to copy bundled model from $assetPath", e)
                return false
            }
        }

        return false
    }

    fun resolveModelPath(): String? {
        val bundledTarget = bundledModelTarget()
        if (bundledTarget.exists()) {
            return bundledTarget.absolutePath
        }

        val externalModel = externalModelCandidate()
        if (externalModel?.exists() == true) {
            return externalModel.absolutePath
        }

        return if (copyBundledModelIfAvailable(bundledTarget)) {
            bundledTarget.absolutePath
        } else {
            null
        }
    }

    @Synchronized
    fun ensureModelLoaded(): Boolean {
        if (modelPtr != 0L) {
            return true
        }

        return try {
            val modelPath = resolveModelPath()
            if (modelPath == null) {
                val externalModelPath = externalModelCandidate()?.absolutePath ?: "external files/models/$MODEL_FILE_NAME"
                Log.e(
                    TAG,
                    "No local model found. Put $MODEL_FILE_NAME at $externalModelPath or bundle it under assets/models/."
                )
                return false
            }
            modelPtr = initModel(modelPath)
            if (modelPtr == 0L) {
                Log.e(TAG, "Failed to load model from $modelPath")
                false
            } else {
                Log.i(TAG, "Model loaded from $modelPath")
                true
            }
        } catch (e: Exception) {
            Log.e(TAG, "Model initialization failed", e)
            false
        }
    }

    fun infer(prompt: String): String {
        if (prompt.isBlank()) {
            return ""
        }
        if (!ensureModelLoaded()) {
            return ""
        }
        return try {
            inferNative(modelPtr, prompt)
        } catch (e: UnsatisfiedLinkError) {
            Log.e(TAG, "Native inference bridge is incomplete", e)
            ""
        } catch (e: Exception) {
            Log.e(TAG, "Native inference failed", e)
            ""
        }
    }

    fun close() {
        if (modelPtr != 0L) {
            releaseModel(modelPtr)
            modelPtr = 0L
        }
    }
}
