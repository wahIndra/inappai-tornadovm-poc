<#
.SYNOPSIS
  Runs the In-App AI Agent on TornadoVM (GPU) or plain JVM (CPU).

.EXAMPLE
  .\run.ps1                # CPU (default): pure-Java inference with the Vector API
  .\run.ps1 gpu            # TornadoVM on the GPU - worth it on a discrete GPU, slower on integrated ones
  .\run.ps1 devices        # list OpenCL devices TornadoVM can see
#>
param(
    [ValidateSet('gpu', 'cpu', 'devices')]
    [string]$Mode = 'cpu',
    [string]$Heap = '6g',
    [string]$DeviceMemory = '4GB'
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot

# Separate from JAVA_HOME on purpose: the system JAVA_HOME usually points to an older JDK.
if (-not $env:JAVA25_HOME) { $env:JAVA25_HOME = Join-Path $HOME '.jdks\jdk-25.0.2' }
if (-not $env:TORNADOVM_HOME) { $env:TORNADOVM_HOME = Join-Path $root '.tools\tornadovm-4.0.1-jdk25-opencl' }
$java = Join-Path $env:JAVA25_HOME 'bin\java.exe'
if (-not (Test-Path $java)) { throw "JDK 25 not found at $env:JAVA25_HOME (set JAVA25_HOME)" }

# Java argfiles treat backslash as an escape inside quotes, so use forward slashes everywhere.
function Slash([string]$p) { $p -replace '\\', '/' }

# The stock `tornado` launcher breaks on paths with spaces, so build our own argfile from the SDK template.
function New-TornadoArgFile {
    $sdk = Slash $env:TORNADOVM_HOME
    $template = Join-Path $env:TORNADOVM_HOME 'tornado-argfile.template'
    if (-not (Test-Path $template)) { throw "TornadoVM SDK not found at $env:TORNADOVM_HOME (set TORNADOVM_HOME)" }

    $lines = Get-Content $template |
        Where-Object { $_.Trim() -and -not $_.Trim().StartsWith('#') } |
        ForEach-Object {
            $line = $_.Trim()
            if ($line.Contains('${TORNADOVM_HOME}')) {
                # "--module-path X" style options become two tokens; quote only the path part.
                $line = (Slash $line).Replace('${TORNADOVM_HOME}', $sdk)
                if ($line -match '^(--[\w-]+) (.+)$') { "$($Matches[1]) `"$($Matches[2])`"" } else { "`"$line`"" }
            } else { $line }
        }

    $argFile = Join-Path $root 'target\tornado.args'
    New-Item -ItemType Directory -Force (Split-Path $argFile) | Out-Null
    Set-Content -Path $argFile -Value $lines -Encoding ascii
    $argFile
}

if ($Mode -eq 'devices') {
    $argFile = New-TornadoArgFile
    & $java "@$argFile" uk.ac.manchester.tornado.drivers.TornadoDeviceQuery verbose
    exit $LASTEXITCODE
}

$classes = Join-Path $root 'target\classes'
$lib = Join-Path $root 'target\lib'
if (-not (Test-Path $lib)) { throw "Build first: mvn -DskipTests package" }
$classpath = "$classes;$lib\*"

$jvmArgs = @("-Xms$Heap", "-Xmx$Heap")
if ($Mode -eq 'gpu') {
    $argFile = New-TornadoArgFile
    $jvmArgs += @("@$argFile", "-Dtornado.device.memory=$DeviceMemory")
    $env:LLM_ON_GPU = 'true'
} else {
    $jvmArgs += @('--enable-preview', '--add-modules', 'jdk.incubator.vector')
    $env:LLM_ON_GPU = 'false'
}

Push-Location $root
try {
    & $java @jvmArgs -cp $classpath id.poc.inappai.InAppAiApplication
} finally {
    Pop-Location
}
