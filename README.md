# Health Workers Assistant

An offline-first Android analytics assistant that lets health workers ask natural-language questions about DHIS2-style data. The application translates questions into SQL, validates and executes the query against a local SQLite database, and returns structured analytics without requiring a cloud-hosted language model.

This is an analytics prototype, not a clinical decision-support system or a substitute for professional medical judgement.

## What it does

- Converts natural-language questions into SQLite queries.
- Uses schema context to produce database-aware SQL.
- Combines deterministic DHIS2 query templates with local GGUF-model inference.
- Runs an automated correction loop when a generated query fails.
- Produces structured analytics payloads for the mobile interface.
- Packages local model inference and data access into an Android application.

## Architecture

```text
React + Capacitor interface
           |
           v
Android native integration
           |
     +-----+------+
     |            |
llama.cpp      Python workflow
local GGUF     PocketFlow + SQLite
inference      text-to-SQL + analytics
```

## Technology

- **Mobile UI:** React, Vite, Capacitor and Recharts
- **Android:** Native Android project with a llama.cpp inference module
- **Workflow:** Python and PocketFlow
- **Data:** SQLite with DHIS2-style analytical data
- **AI:** Local quantised GGUF models for text-to-SQL and answer generation

## Repository layout

```text
Frontend/                         React + Capacitor application
Frontend/android/                 Android project
Frontend/android/llama-engine/    Native local-inference module
pocketflow-text2sql/              Python text-to-SQL and analytics workflow
```

## First-time setup

### Install frontend dependencies

```powershell
cd Frontend
npm install
cd ..
```

### Bootstrap llama.cpp

```powershell
.\Frontend\scripts\bootstrap-llama-cpp.ps1
```

### Install the optional local Python runtime

```powershell
cd pocketflow-text2sql
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
cd ..
```

The Android build expects the required SQLite databases and GGUF model assets inside `pocketflow-text2sql/`. See the detailed workflow documentation in [pocketflow-text2sql/README.md](pocketflow-text2sql/README.md).

## Verify the text-to-SQL workflow

```powershell
cd pocketflow-text2sql
python main.py
```

A custom natural-language question can be passed after the script name:

```powershell
python main.py "How many organisation units are there at each hierarchy level?"
```

## Build the Android application

Create a local signing configuration if one does not already exist:

```powershell
.\Frontend\scripts\create-release-keystore.ps1
```

Build the release APK:

```powershell
.\Frontend\scripts\build-android.ps1 -Configuration Release
```

The release APK is written to:

```text
Frontend/android/app/build/outputs/apk/release/app-release.apk
```

Signing keys and local credentials must never be committed.
