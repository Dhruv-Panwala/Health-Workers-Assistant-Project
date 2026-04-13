$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$repoRoot = Split-Path -Parent $projectRoot
$vendorDir = Join-Path $repoRoot ".vendor"
$llamaDir = Join-Path $vendorDir "llama.cpp"

if (-not (Test-Path $vendorDir)) {
    New-Item -ItemType Directory -Force -Path $vendorDir | Out-Null
}

if (-not (Test-Path (Join-Path $llamaDir "CMakeLists.txt"))) {
    git clone --depth 1 https://github.com/ggml-org/llama.cpp.git $llamaDir
}
