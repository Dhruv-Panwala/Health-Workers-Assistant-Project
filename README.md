# Chat Assistant Restart for Android

This repository contains:

- `Frontend/`: the React + Capacitor Android app
- `Frontend/android/llama-engine/`: the native Android inference module
- `pocketflow-text2sql/`: the bundled Python runtime, SQLite databases, and GGUF model files used by the app

The current build flow packages the Android app from `Frontend/` and copies runtime assets from `pocketflow-text2sql/` during the Gradle build.

## First-Time Setup

### 1. Clone the repository

### 2. Install frontend dependencies

In powershell run commands:
1. cd Frontend
2. npm install
3. cd ..


### 3. Bootstrap `llama.cpp`

This creates `.vendor/llama.cpp`, which is required by the Android native module.

In powershell run command:
.\Frontend\scripts\bootstrap-llama-cpp.ps1

### 4. Verify bundled runtime assets exist

The Android build expects these files inside `pocketflow-text2sql/`:
- `dhis2.sqlite`
- `dhis2_analytics.db`
- `llama-3-sqlcoder-0.5b.gguf`
- `CURE-MED-1.5B.i1-Q4_K_M.gguf`

If any of them are missing, the Android build will fail during `syncBundledAssets`.

### 5. Optional: install Python dependencies for local runtime testing

This is useful if you want to run the PocketFlow backend locally outside the APK build.

In powershell run commands:
1. cd pocketflow-text2sql
2. python -m venv .venv
3. .\.venv\Scripts\Activate.ps1
4. pip install -r requirements.txt
5. cd ..


## Local Verification

To verify the Python runtime locally:

In powershell run command:
1. cd pocketflow-text2sql
2. python main.py
3. cd ..

## Generate a Release APK

### 1. Create a signing keystore

If you do not already have one:

In powershell run command: 
.\Frontend\scripts\create-release-keystore.ps1

This creates:

- `Frontend/android/keystore.properties`
- `Frontend/uliza-release.keystore`

You can also inspect the template at "Frontend/android/keystore.properties.example"

### 2. Build the release APK

In powershell run command: 
.\Frontend\scripts\build-android.ps1 -Configuration Release


Release APK will be created in : "Frontend\android\app\build\outputs\apk\release\app-release.apk"

## Open the Android Project in Android Studio

If you want to inspect or build from Android Studio:

In powershell run command:
1. cd Frontend
2. npm run cap -- open android

## Output Paths

- Debug APK: `Frontend\android\app\build\outputs\apk\debug\app-debug.apk`
- Release APK: `Frontend\android\app\build\outputs\apk\release\app-release.apk`

