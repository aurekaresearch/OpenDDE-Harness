# OpenDDE Harness installer for native Windows PowerShell.
#
# Private repository: authenticate Git, clone the repository, then run
# .\install.ps1 from the checkout root.
#
# Goal: a clean Windows machine ends up able to run `ddeharness` / `ddeharness tui`
# without admin rights. The script is idempotent: it reuses existing tools when
# available and only fills the gaps:
#   1. uv            (Python toolchain + package manager)
#   2. Node.js >= 22 (TUI runtime; installed privately if the system lacks it)
#   3. opendde         (installed as a global uv tool)

$ErrorActionPreference = "Stop"

$MinNodeMajor = 22
$OpenDDEHarnessHome = if ($env:OPENDDE_HARNESS_HOME) { $env:OPENDDE_HARNESS_HOME } else { Join-Path $HOME ".opendde_harness" }
$NodeRuntimeDir = Join-Path $OpenDDEHarnessHome "runtime"

function Write-Info([string]$Message) {
    Write-Host ">" $Message -ForegroundColor Cyan
}

function Write-Ok([string]$Message) {
    Write-Host "OK" $Message -ForegroundColor Green
}

function Write-Warn([string]$Message) {
    Write-Warning $Message
}

function Fail([string]$Message) {
    Write-Error $Message
    exit 1
}

function Add-ProcessPath([string]$PathToAdd) {
    if (-not $PathToAdd) { return }
    if (-not (Test-Path $PathToAdd)) { return }
    $parts = $env:PATH -split ';'
    if ($parts -notcontains $PathToAdd) {
        $env:PATH = "$PathToAdd;$env:PATH"
    }
}

function Find-Uv {
    $cmd = Get-Command uv -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }

    $candidates = @(
        (Join-Path $HOME ".local\bin\uv.exe"),
        (Join-Path $env:USERPROFILE ".local\bin\uv.exe")
    )
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) { return $candidate }
    }
    return $null
}

function Ensure-Uv {
    $uv = Find-Uv
    if ($uv) {
        Write-Ok "uv is installed ($(& $uv --version))"
        Add-ProcessPath (Split-Path $uv -Parent)
        return $uv
    }

    Write-Info "uv not found; installing..."
    Invoke-Expression (Invoke-RestMethod "https://astral.sh/uv/install.ps1")
    $uv = Find-Uv
    if (-not $uv) {
        Fail "uv was installed but is still not available. Check PATH (expected ~/.local/bin)."
    }
    Add-ProcessPath (Split-Path $uv -Parent)
    Write-Ok "uv installed"
    return $uv
}

function Get-NodeArch {
    switch ($env:PROCESSOR_ARCHITECTURE) {
        "ARM64" { return "arm64" }
        "AMD64" { return "x64" }
        default { Fail "Unsupported Windows architecture: $env:PROCESSOR_ARCHITECTURE" }
    }
}

function Test-NodeOk([string]$NodePath) {
    if (-not $NodePath) { return $false }
    if (-not (Test-Path $NodePath)) { return $false }
    try {
        $version = (& $NodePath --version).Trim()
        $major = [int](($version.TrimStart("v") -split "\.")[0])
        return $major -ge $MinNodeMajor
    } catch {
        return $false
    }
}

function Find-PrivateNode {
    $candidates = @()
    $direct = Join-Path $NodeRuntimeDir "node\node.exe"
    $directBin = Join-Path $NodeRuntimeDir "node\bin\node.exe"
    if (Test-Path $direct) { $candidates += $direct }
    if (Test-Path $directBin) { $candidates += $directBin }
    if (Test-Path $NodeRuntimeDir) {
        $candidates += Get-ChildItem $NodeRuntimeDir -Directory -Filter "node-v22*" -ErrorAction SilentlyContinue |
            ForEach-Object {
                @(
                    (Join-Path $_.FullName "node.exe"),
                    (Join-Path $_.FullName "bin\node.exe")
                )
            }
    }
    foreach ($candidate in $candidates) {
        if (Test-NodeOk $candidate) { return $candidate }
    }
    return $null
}

function Test-SourceCheckout([string]$Dir) {
    $pyproject = Join-Path $Dir "pyproject.toml"
    return (Test-Path $pyproject) -and (Select-String -Path $pyproject -Pattern '^name = "opendde-harness"' -Quiet)
}

# The pinned Node.js release, read from the client so the installer and
# `ddeharness tui` (which provisions Node for wheel installs) agree on it.
function Get-PinnedNodeVersion([string]$Dir) {
    $file = Join-Path $Dir "opendde_harness\cli\node_runtime.py"
    $match = Select-String -Path $file -Pattern '^NODE_VERSION = "([0-9.]+)"' | Select-Object -First 1
    if (-not $match) { Fail "Pinned Node.js version not found in $file" }
    return "v" + $match.Matches[0].Groups[1].Value
}

function Ensure-Node([string]$SourceDir) {
    $systemNode = Get-Command node -ErrorAction SilentlyContinue
    if ($systemNode -and (Test-NodeOk $systemNode.Source)) {
        Write-Ok "Node.js meets requirements ($(& $systemNode.Source --version))"
        return $systemNode.Source
    }

    $privateNode = Find-PrivateNode
    if ($privateNode) {
        Write-Ok "Existing OpenDDE Harness private Node found ($privateNode)"
        Add-ProcessPath (Split-Path $privateNode -Parent)
        return $privateNode
    }

    Write-Info "Node.js >= $MinNodeMajor not found; downloading private runtime..."
    $arch = Get-NodeArch
    $version = Get-PinnedNodeVersion $SourceDir
    $pkg = "node-$version-win-$arch"
    $url = "https://nodejs.org/dist/$version/$pkg.zip"
    $tmp = Join-Path ([IO.Path]::GetTempPath()) ("opendde-node-" + [guid]::NewGuid().ToString("N"))
    $zipPath = Join-Path $tmp "node.zip"

    New-Item -ItemType Directory -Path $tmp -Force | Out-Null
    New-Item -ItemType Directory -Path $NodeRuntimeDir -Force | Out-Null

    try {
        Write-Info "  $url"
        Invoke-WebRequest $url -OutFile $zipPath

        try {
            $sums = (Invoke-WebRequest "https://nodejs.org/dist/$version/SHASUMS256.txt").Content
            $line = ($sums -split "`n") | Where-Object { $_ -match "\s+$([regex]::Escape("$pkg.zip"))$" } | Select-Object -First 1
            if ($line) {
                $expected = (($line.Trim()) -split "\s+")[0].ToLowerInvariant()
                $actual = (Get-FileHash $zipPath -Algorithm SHA256).Hash.ToLowerInvariant()
                if ($expected -ne $actual) {
                    Fail "Node checksum mismatch (expected $expected, got $actual)."
                }
                Write-Ok "Node zip SHA256 verified"
            } else {
                Write-Warn "SHASUMS256.txt did not list $pkg.zip; skipping checksum verification"
            }
        } catch {
            Write-Warn "Could not verify Node checksum; continuing"
        }

        Expand-Archive $zipPath -DestinationPath $tmp -Force
        $src = Join-Path $tmp $pkg
        $dest = Join-Path $NodeRuntimeDir $pkg
        if (Test-Path $dest) { Remove-Item $dest -Recurse -Force }
        Move-Item $src $dest

        $node = Join-Path $dest "node.exe"
        if (-not (Test-NodeOk $node)) {
            Fail "Downloaded Node runtime is not usable on this machine."
        }
        Add-ProcessPath $dest
        Write-Ok "Node private runtime ready: $dest"
        return $node
    } finally {
        if (Test-Path $tmp) { Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue }
    }
}

# Reads the latest stable tag off the release page redirect. The GitHub API caps
# unauthenticated callers at 60 requests/hour per IP, which a shared egress can
# exhaust; the release page carries no API quota. Returns "" when the redirect is
# missing or does not name a stable tag, so the caller can fail with its own message.
function Resolve-OpenDDEHarnessLatestVersion {
    $target = ""
    try {
        $response = Invoke-WebRequest "https://github.com/aurekaresearch/OpenDDE-Harness-beta/releases/latest" -MaximumRedirection 0 -UseBasicParsing -ErrorAction Stop
        $target = [string]$response.Headers.Location
    } catch {
        # Windows PowerShell raises on an unfollowed redirect; the Location header
        # still rides on the exception's response.
        $failed = $_.Exception.Response
        if ($failed) {
            try { $target = [string]$failed.Headers.Location } catch { $target = "" }
            if (-not $target) {
                try { $target = [string]$failed.Headers.GetValues("Location")[0] } catch { $target = "" }
            }
        }
    }
    if ($target -match "^https://github\.com/aurekaresearch/OpenDDE-Harness-beta/releases/tag/v([0-9]+\.[0-9]+\.[0-9]+)$") {
        return $Matches[1]
    }
    return ""
}

function Resolve-OpenDDEHarnessWheel {
    if ($env:OPENDDE_HARNESS_WHEEL_URL) { return $env:OPENDDE_HARNESS_WHEEL_URL }
    Write-Info "Resolving the latest OpenDDE Harness release from GitHub..."
    try {
        $release = Invoke-RestMethod "https://api.github.com/repos/aurekaresearch/OpenDDE-Harness-beta/releases/latest" -Headers @{ "User-Agent" = "opendde-installer" }
        $asset = $release.assets | Where-Object { $_.browser_download_url -match "/opendde_harness-[^/]+\.whl$" } | Select-Object -First 1
        if ($asset) { return $asset.browser_download_url }
        Write-Warn "GitHub API returned no release wheel; falling back to the release page."
    } catch {
        Write-Warn "GitHub API lookup failed ($($_.Exception.Message)); falling back to the release page."
    }
    $version = Resolve-OpenDDEHarnessLatestVersion
    if (-not $version) {
        Fail "Could not resolve the latest OpenDDE Harness release wheel from GitHub. Retry later, or set OPENDDE_HARNESS_WHEEL_URL to a wheel URL."
    }
    return "https://github.com/aurekaresearch/OpenDDE-Harness-beta/releases/download/v$version/opendde_harness-$version-py3-none-any.whl"
}

function Resolve-OpenDDEHarnessConstraints([string]$WheelUrl) {
    # Derive the locked-constraints URL from the wheel URL (same release dir) so
    # the constraints always match the wheel being installed -- including when
    # OPENDDE_HARNESS_WHEEL_URL pins an older wheel. Returns a local temp-file path, or
    # $null when the asset is absent (release predates it) or the download fails,
    # so the installer degrades to an unconstrained install rather than failing.
    $url = $env:OPENDDE_HARNESS_CONSTRAINTS_URL
    if (-not $url) {
        if ($WheelUrl -notmatch "/[^/]+\.whl$") { return $null }
        $url = $WheelUrl -replace "/[^/]+\.whl$", "/opendde-harness-constraints.txt"
    }
    $dest = Join-Path ([IO.Path]::GetTempPath()) ("opendde-harness-constraints-" + [guid]::NewGuid().ToString("N") + ".txt")
    try {
        Invoke-WebRequest $url -OutFile $dest
    } catch {
        Write-Warn "Could not download locked constraints; installing without version pinning."
        return $null
    }
    return $dest
}

function Install-OpenDDEHarness([string]$UvPath, [string]$NodePath) {
    $scriptDir = if ($PSScriptRoot) { $PSScriptRoot } else { (Get-Location).Path }
    if (Test-SourceCheckout $scriptDir) {
        Write-Info "Building a standalone installation from source: $scriptDir"
        $entry = Join-Path $scriptDir "ui-tui\dist\entry.js"
        $nodeDir = Split-Path $NodePath -Parent
        Add-ProcessPath $nodeDir
        $npm = Get-Command npm -ErrorAction SilentlyContinue
        if (-not $npm) { Fail "A complete Node runtime with npm is required to build the TUI." }
        Write-Info "Building TUI bundle (ui-tui/dist/entry.js)..."
        Push-Location (Join-Path $scriptDir "ui-tui")
        try {
            & $npm.Source ci
            if ($LASTEXITCODE -ne 0) { Fail "TUI dependency installation failed." }
            & $npm.Source run build
            if ($LASTEXITCODE -ne 0) { Fail "TUI build failed." }
        } finally {
            Pop-Location
        }
        if (-not (Test-Path $entry)) { Fail "TUI build did not produce entry.js." }
        # Pin to the locked dependency set so an install matches what we test.
        $constraints = Join-Path ([IO.Path]::GetTempPath()) ("opendde-harness-constraints-" + [guid]::NewGuid().ToString("N") + ".txt")
        & $UvPath export --directory "$scriptDir" --frozen --no-hashes --no-emit-project -o "$constraints" | Out-Null
        if ($LASTEXITCODE -ne 0) { Fail "Dependency constraints export failed." }
        & $UvPath tool install --force --python 3.12 -c "$constraints" "$scriptDir"
        if ($LASTEXITCODE -ne 0) { Fail "OpenDDE Harness install failed." }
    } else {
        $wheelUrl = Resolve-OpenDDEHarnessWheel
        $constraints = Resolve-OpenDDEHarnessConstraints $wheelUrl
        if ($constraints) {
            $cArgs = @("-c", $constraints)
        } else {
            Write-Warn "Release has no locked-constraints asset; installing without version pinning."
            $cArgs = @()
        }
        Write-Info "  installing $wheelUrl"
        & $UvPath tool install --force --python 3.12 @cArgs $wheelUrl
        if ($LASTEXITCODE -ne 0) { Fail "OpenDDE Harness install failed." }
    }
    & $UvPath tool update-shell | Out-Null
    $installedBin = (& $UvPath tool dir --bin).Trim()
    $entryCommand = Join-Path $installedBin "ddeharness.exe"
    if (-not (Test-Path $entryCommand)) { Fail "Installed package did not create ddeharness.exe." }
    & $entryCommand --version
    if ($LASTEXITCODE -ne 0) { Fail "Installed CLI failed its startup check." }
    # A wheel install needs no Node up front: this check installs a private Node.js
    # runtime when the system has none (set OPENDDE_HARNESS_NO_NODE_INSTALL=1 to forbid it).
    & $entryCommand tui --check
    if ($LASTEXITCODE -ne 0) { Fail "Installed TUI failed its startup check." }
    Write-Ok "OpenDDE Harness installed"
}

function Main {
    # Read before installing so the closing hint can tell a first run from an
    # upgrade; the install itself never writes config.json (the wizard does).
    $hadConfig = Test-Path (Join-Path $OpenDDEHarnessHome "config.json")

    $uv = Ensure-Uv
    # Building the TUI from source needs Node with npm before the client exists;
    # a wheel install lets `ddeharness tui --check` provision Node itself.
    $scriptDir = if ($PSScriptRoot) { $PSScriptRoot } else { (Get-Location).Path }
    $node = if (Test-SourceCheckout $scriptDir) { Ensure-Node $scriptDir } else { "" }
    Install-OpenDDEHarness $uv $node

    $toolBin = Join-Path $HOME ".local\bin"
    Add-ProcessPath $toolBin

    Write-Host ""
    if ($hadConfig) {
        Write-Ok "OpenDDE Harness updated. Your config in $OpenDDEHarnessHome is unchanged."
        Write-Host ""
        Write-Host "    ddeharness    # continue where you left off"
        Write-Host ""
        Write-Host "  tip: next time you can upgrade in place with 'ddeharness upgrade'"
        Write-Host ""
    } else {
        Write-Ok "All set. Open a new PowerShell window, or continue in this one, then run:"
        Write-Host ""
        Write-Host "    ddeharness    # sets you up on first run, then opens the TUI"
        Write-Host ""
    }
    if (($env:PATH -split ';') -notcontains $toolBin) {
        Write-Warn "Current PATH does not include $toolBin. Restart PowerShell if 'ddeharness' is not found."
    }
}

Main
