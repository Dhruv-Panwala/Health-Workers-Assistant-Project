package com.healthworker.assistant

import android.content.Context
import java.io.File
import java.io.FileOutputStream

object DatabaseInstaller {

    fun installDatabase(context: Context, dbName: String = "dhis2.sqlite") {
        val dbDir = File(context.filesDir, "databases")
        if (!dbDir.exists()) dbDir.mkdirs()

        val dbFile = File(dbDir, dbName)

        if (dbFile.exists()) return  // already installed

        context.assets.open("databases/$dbName").use { input ->
            FileOutputStream(dbFile).use { output ->
                input.copyTo(output)
            }
        }
    }
}