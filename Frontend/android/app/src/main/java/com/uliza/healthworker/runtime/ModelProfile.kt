package com.uliza.healthworker.runtime

enum class ModelProfile(
    val wireName: String,
    val assetFileName: String,
    val defaultTemperature: Float,
) {
    SQL(
        wireName = "sql",
        assetFileName = "llama-3-sqlcoder-0.5b.gguf",
        defaultTemperature = 0.0f,
    ),
    CHAT(
        wireName = "chat",
        assetFileName = "CURE-MED-1.5B.i1-Q4_K_M.gguf",
        defaultTemperature = 0.2f,
    );

    companion object {
        fun fromWireName(value: String): ModelProfile =
            entries.firstOrNull { it.wireName == value }
                ?: throw IllegalArgumentException("Unknown model profile: $value")
    }
}
