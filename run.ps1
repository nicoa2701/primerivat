$ErrorActionPreference = 'Stop'
$ScriptRoot = $PSScriptRoot
Set-Location -LiteralPath $ScriptRoot

# 1. Build release
cargo build --release
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# 2. Discover binary via cargo metadata (handles workspace, default-run, custom target dir).
$metaJson = cargo metadata --format-version 1 --no-deps
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$meta = $metaJson | ConvertFrom-Json

# 3. Pick the right package
$pkgs = @($meta.packages)
if ($pkgs.Count -eq 0) { Write-Error 'No package in cargo metadata'; exit 1 }
$pkg = $pkgs[0]
# In a workspace, prefer the package matching workspace_default_members
$defaultMembers = $meta.workspace_default_members
if ($defaultMembers -and @($defaultMembers).Count -gt 0) {
    $match = $pkgs | Where-Object { @($defaultMembers) -contains $_.id } | Select-Object -First 1
    if ($match) { $pkg = $match }
}

# 4. Pick the right [[bin]] target
$bins = @($pkg.targets | Where-Object { $_.kind -contains 'bin' })
if ($bins.Count -eq 0) { Write-Error "No [[bin]] target in package '$($pkg.name)'"; exit 1 }

if ($pkg.default_run) {
    $bin = $bins | Where-Object { $_.name -eq $pkg.default_run } | Select-Object -First 1
    if (-not $bin) {
        Write-Error "[package] default-run = '$($pkg.default_run)' not found among [[bin]] targets"
        exit 1
    }
} elseif ($bins.Count -eq 1) {
    $bin = $bins[0]
} else {
    Write-Warning ("Multiple [[bin]] targets; using '{0}'. Set [package] default-run in Cargo.toml to disambiguate." -f $bins[0].name)
    $bin = $bins[0]
}

# 5. Build exe path (use cargo's target_directory; add .exe on Windows)
$targetDir = $meta.target_directory
$exeSuffix = if ($env:OS -eq 'Windows_NT') { '.exe' } else { '' }
$exe = Join-Path $targetDir ('release\' + $bin.name + $exeSuffix)
if (-not (Test-Path -LiteralPath $exe)) { Write-Error "Executable not found: $exe"; exit 1 }

# 6. Forward args, propagate exit code
& $exe @args
exit $LASTEXITCODE
