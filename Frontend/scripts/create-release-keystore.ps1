param(
    [string]$Alias = "uliza",
    [string]$StoreFile = "../../uliza-release.keystore",
    [string]$StorePassword = "uliza123",
    [string]$KeyPassword = "uliza123"
)

$ErrorActionPreference = "Stop"
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$androidDir = Join-Path (Split-Path -Parent $scriptRoot) "android"
$storePath = [System.IO.Path]::GetFullPath((Join-Path $androidDir $StoreFile))
$keytool = Join-Path $env:JAVA_HOME "bin\keytool.exe"

if (-not (Test-Path $keytool)) {
    throw "keytool.exe was not found. Set JAVA_HOME to a JDK installation first."
}

if (-not (Test-Path $storePath)) {
    & $keytool -genkeypair `
        -alias $Alias `
        -keyalg RSA `
        -keysize 2048 `
        -validity 3650 `
        -storetype PKCS12 `
        -keystore $storePath `
        -storepass $StorePassword `
        -keypass $KeyPassword `
        -dname "CN=Uliza Health Worker, OU=Engineering, O=Uliza, L=London, S=London, C=GB"
}

@"
storeFile=$StoreFile
storePassword=$StorePassword
keyAlias=$Alias
keyPassword=$KeyPassword
"@ | Set-Content -Encoding ASCII (Join-Path $androidDir "keystore.properties")
