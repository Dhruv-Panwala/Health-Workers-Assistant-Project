param(
    [ValidateSet("Debug", "Release")]
    [string]$Configuration = "Debug"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$androidDir = Join-Path $projectRoot "android"
$sdkRoot = Join-Path $env:LOCALAPPDATA "Android\Sdk"
$vendorRoot = Join-Path (Split-Path -Parent $projectRoot) ".vendor\llama.cpp"

$env:ANDROID_HOME = $sdkRoot
$env:ANDROID_SDK_ROOT = $sdkRoot

if (-not (Test-Path (Join-Path $vendorRoot "CMakeLists.txt"))) {
    throw "Missing llama.cpp source checkout at '$vendorRoot'. Run scripts/bootstrap-llama-cpp.ps1 first."
}

Push-Location $projectRoot
try {
    npm run build
    npm run cap -- sync android

    Push-Location $androidDir
    try {
        .\gradlew.bat "assemble$Configuration"
    }
    finally {
        Pop-Location
    }
}
finally {
    Pop-Location
}
