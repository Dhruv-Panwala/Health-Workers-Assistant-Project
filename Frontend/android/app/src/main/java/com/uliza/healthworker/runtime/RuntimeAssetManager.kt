package com.uliza.healthworker.runtime

import android.content.Context
import org.json.JSONObject
import java.io.File
import java.security.MessageDigest

data class RuntimePaths(
    val assetsDir: File,
    val activeDbPath: String,
    val sqlModelPath: String,
    val chatModelPath: String,
)

data class PrepareResult(
    val ready: Boolean,
    val copiedBytes: Long,
    val totalBytes: Long,
    val activeDb: String,
)

data class RuntimeStatus(
    val assetsReady: Boolean,
    val activeDb: String,
)

data class RuntimeBundleInfo(
    val totalBytes: Long,
)

data class BundledAsset(
    val name: String,
    val sizeBytes: Long,
    val sha256: String,
)

object RuntimeAssetManager {
    private data class PreparedRuntimeManifest(
        val manifestText: String,
        val runtimePaths: RuntimePaths,
        val expectedFiles: List<File>,
    )

    private val ASSET_ROOT_CANDIDATES = listOf("bundled", "")
    private const val MANIFEST_NAME = "runtime-manifest.json"
    private const val DEFAULT_ACTIVE_DB = "dhis2.sqlite"

    @Volatile
    private var cachedAssetRoot: String? = null

    @Volatile
    private var cachedManifestText: String? = null

    private val prepareLock = Any()

    fun prepare(context: Context): PrepareResult {
        synchronized(prepareLock) {
            val runtimeDir = runtimeDir(context)
            runtimeDir.mkdirs()

            val manifestText = assetManifestText(context)
            val manifestJson = JSONObject(manifestText)
            val assets = manifestAssets(manifestJson)
            val totalBytes = assets.sumOf { it.sizeBytes }
            val activeDbName = manifestJson.optString("activeDb", DEFAULT_ACTIVE_DB)
            var copiedBytes = 0L
            val assetRoot = resolveAssetRoot(context)

            assets.forEach { asset ->
                val destination = File(runtimeDir, asset.name)
                val needsCopy = !destination.exists() || destination.length() != asset.sizeBytes || checksum(destination) != asset.sha256
                if (needsCopy) {
                    destination.parentFile?.mkdirs()
                    context.assets.open(assetPath(assetRoot, asset.name)).use { input ->
                        destination.outputStream().use { output ->
                            input.copyTo(output, DEFAULT_BUFFER_SIZE * 8)
                        }
                    }
                    if (checksum(destination) != asset.sha256) {
                        destination.delete()
                        throw IllegalStateException("Checksum verification failed for ${asset.name}")
                    }
                    copiedBytes += destination.length()
                }
            }

            localManifestFile(context).writeText(manifestText)

            return PrepareResult(
                ready = true,
                copiedBytes = copiedBytes,
                totalBytes = totalBytes,
                activeDb = File(runtimeDir, activeDbName).absolutePath,
            )
        }
    }

    fun status(context: Context): RuntimeStatus {
        val preparedManifest = readPreparedManifest(context)
        if (preparedManifest == null) {
            return RuntimeStatus(
                assetsReady = false,
                activeDb = File(runtimeDir(context), DEFAULT_ACTIVE_DB).absolutePath,
            )
        }

        val manifestMatches = preparedManifest.manifestText == assetManifestText(context)
        val filesExist = preparedManifest.expectedFiles.all { it.exists() }

        return RuntimeStatus(
            assetsReady = manifestMatches && filesExist,
            activeDb = preparedManifest.runtimePaths.activeDbPath,
        )
    }

    fun bundleInfo(context: Context): RuntimeBundleInfo {
        val manifestJson = manifestJson(context)
        val totalBytes = manifestAssets(manifestJson).sumOf { it.sizeBytes }
        return RuntimeBundleInfo(totalBytes = totalBytes)
    }

    fun resolvePreparedPaths(context: Context): RuntimePaths {
        val preparedManifest = ensurePreparedManifest(context)
        return preparedManifest.runtimePaths
    }

    private fun runtimeDir(context: Context) = File(context.filesDir, "uliza-runtime")

    private fun localManifestFile(context: Context) = File(runtimeDir(context), MANIFEST_NAME)

    private fun assetManifestText(context: Context): String =
        cachedManifestText ?: synchronized(this) {
            cachedManifestText
                ?: context.assets
                    .open(assetPath(resolveAssetRoot(context), MANIFEST_NAME))
                    .bufferedReader()
                    .use { it.readText() }
                    .also { cachedManifestText = it }
        }

    private fun manifestJson(context: Context): JSONObject = JSONObject(assetManifestText(context))

    private fun resolveAssetRoot(context: Context): String =
        cachedAssetRoot ?: synchronized(this) {
            cachedAssetRoot ?: ASSET_ROOT_CANDIDATES.firstOrNull { root ->
                try {
                    context.assets.open(assetPath(root, MANIFEST_NAME)).use { }
                    true
                } catch (_: Exception) {
                    false
                }
            }?.also { cachedAssetRoot = it }
                ?: throw IllegalStateException("runtime-manifest.json was not found in APK assets")
        }

    private fun assetPath(root: String, name: String): String =
        if (root.isBlank()) name else "$root/$name"

    private fun manifestAssets(manifestJson: JSONObject): List<BundledAsset> {
        val assets = manifestJson.getJSONArray("assets")
        return buildList {
            for (index in 0 until assets.length()) {
                val entry = assets.getJSONObject(index)
                add(
                    BundledAsset(
                        name = entry.getString("name"),
                        sizeBytes = entry.getLong("sizeBytes"),
                        sha256 = entry.getString("sha256"),
                    )
                )
            }
        }
    }

    private fun ensurePreparedManifest(context: Context): PreparedRuntimeManifest {
        val assetManifestText = assetManifestText(context)
        val existing = readPreparedManifest(context)
        val manifestIsCurrent = existing?.manifestText == assetManifestText
        val missingFiles = existing?.expectedFiles?.filterNot { it.exists() }.orEmpty()

        if (existing == null || !manifestIsCurrent || missingFiles.isNotEmpty()) {
            prepare(context)
        }

        val refreshed = readPreparedManifest(context)
            ?: throw IllegalStateException("Prepared runtime manifest is missing or invalid")
        if (refreshed.manifestText != assetManifestText) {
            throw IllegalStateException("Bundled runtime assets are not prepared yet")
        }

        val unresolvedMissingFiles = refreshed.expectedFiles.filterNot { it.exists() }
        if (unresolvedMissingFiles.isNotEmpty()) {
            throw IllegalStateException(
                "Prepared runtime assets are missing: ${unresolvedMissingFiles.joinToString(", ") { it.absolutePath }}"
            )
        }

        return refreshed
    }

    private fun readPreparedManifest(context: Context): PreparedRuntimeManifest? {
        val manifestFile = localManifestFile(context)
        if (!manifestFile.exists()) {
            return null
        }

        return try {
            val manifestText = manifestFile.readText()
            val manifestJson = JSONObject(manifestText)
            val runtimeDir = runtimeDir(context)
            PreparedRuntimeManifest(
                manifestText = manifestText,
                runtimePaths = RuntimePaths(
                    assetsDir = runtimeDir,
                    activeDbPath = File(runtimeDir, manifestJson.optString("activeDb", DEFAULT_ACTIVE_DB)).absolutePath,
                    sqlModelPath = File(runtimeDir, ModelProfile.SQL.assetFileName).absolutePath,
                    chatModelPath = File(runtimeDir, ModelProfile.CHAT.assetFileName).absolutePath,
                ),
                expectedFiles = manifestAssets(manifestJson).map { asset -> File(runtimeDir, asset.name) },
            )
        } catch (_: Exception) {
            null
        }
    }

    private fun checksum(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { stream ->
            val buffer = ByteArray(DEFAULT_BUFFER_SIZE * 8)
            while (true) {
                val read = stream.read(buffer)
                if (read <= 0) {
                    break
                }
                digest.update(buffer, 0, read)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }
}
