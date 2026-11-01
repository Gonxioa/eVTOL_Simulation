$ErrorActionPreference = 'Stop'
$workspacePath = 'C:\Users\lenovo\Desktop\eVTOL_Simulation'
$auditPath = Join-Path $workspacePath 'maintenance\archive_20261030'
$receipt = Get-Content -LiteralPath (Join-Path $auditPath 'verified_assets.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$preflight = Get-Content -LiteralPath (Join-Path $auditPath 'deletion_preflight.json') -Raw -Encoding UTF8 | ConvertFrom-Json
if (-not $receipt.complete -or -not $receipt.server_requeried -or -not $preflight.remote_archives_verified) { throw 'Remote archives have not been verified.' }
$workspaceAbsolute = (Resolve-Path -LiteralPath $workspacePath).ProviderPath.TrimEnd('\')
$explicitExternalPaths = @(
    'C:\Users\lenovo\.codex\worktrees\hex-audit-control\eVTOL_Simulation',
    'C:\Users\lenovo\AppData\Local\Temp\evtol_archive_20261030',
    'C:\Users\lenovo\AppData\Local\Temp\evtol_current_verify_20261030'
)
$completed = [System.Collections.Generic.List[object]]::new()
$deletedLogPath = Join-Path $auditPath 'deleted_targets.json'
if (Test-Path -LiteralPath $deletedLogPath) {
    foreach ($previous in @(Get-Content -LiteralPath $deletedLogPath -Raw -Encoding UTF8 | ConvertFrom-Json)) {
        $completed.Add($previous)
    }
}
$protectedNames = @('.git','RflyMAD多旋翼故障数据集','UAV-FD','轴承数据集','模型','docs','outputs','output_uavfd_f550')
foreach ($target in $preflight.targets) {
    if (-not (Test-Path -LiteralPath $target.path)) { continue }
    $resolvedTarget = (Resolve-Path -LiteralPath $target.path).ProviderPath.TrimEnd('\')
    $withinWorkspace = $resolvedTarget.StartsWith($workspaceAbsolute + '\', [System.StringComparison]::OrdinalIgnoreCase)
    if (-not $withinWorkspace -and $resolvedTarget -notin $explicitExternalPaths) { throw "Target escaped permitted roots: $resolvedTarget" }
    if ($resolvedTarget -eq $workspaceAbsolute -or $resolvedTarget -eq (Join-Path $workspaceAbsolute '.git')) { throw 'Refusing to remove the workspace or Git database.' }
    foreach ($protectedName in $protectedNames) {
        $protectedPath = Join-Path $workspaceAbsolute $protectedName
        if ($resolvedTarget.Equals($protectedPath,[System.StringComparison]::OrdinalIgnoreCase) -or $resolvedTarget.StartsWith($protectedPath + '\',[System.StringComparison]::OrdinalIgnoreCase)) { throw "Refusing a protected resource: $resolvedTarget" }
    }
    $item = Get-Item -LiteralPath $resolvedTarget -Force
    if ($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) { throw "Refusing a reparse point: $resolvedTarget" }
    if ($target.kind -eq 'old_worktree') {
        & git -C $workspaceAbsolute worktree remove --force $resolvedTarget
        if ($LASTEXITCODE -ne 0) { throw 'Git worktree removal failed.' }
    } else {
        Remove-Item -LiteralPath $resolvedTarget -Recurse -Force
    }
    $completed.Add([pscustomobject]@{ path=$resolvedTarget; kind=$target.kind; files=$target.files; bytes=$target.bytes })
    $completed | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $deletedLogPath -Encoding UTF8
}
$sourceBytes = ($completed | Measure-Object -Property bytes -Sum).Sum
[pscustomobject]@{ removedTargets=$completed.Count; removedSourceBytes=$sourceBytes; archiveUrl=$receipt.release_url } | ConvertTo-Json
