param([Parameter(Mandatory=$true)][string]$WorkspaceRoot)
$ErrorActionPreference = 'Stop'
$layoutRoot = (Resolve-Path -LiteralPath $WorkspaceRoot).Path.TrimEnd('\')
if (-not (Test-Path -LiteralPath (Join-Path $layoutRoot '.git'))) { throw 'Expected Git workspace root' }
$layoutMap = [ordered]@{
    'src'='code/src'; 'tests'='code/tests'; 'tools'='code/tools';
    'configs'='code/configs'; 'research_versions'='code/research_versions';
    'runs'='code/runs'; 'data'='code/data'; '.venv'='code/.venv';
    '.pytest_cache'='code/.pytest_cache'; 'pyproject.toml'='code/pyproject.toml';
    'uv.lock'='code/uv.lock'; '.python-version'='code/.python-version';
    'docs'='materials/docs'; 'literature'='materials/literature';
    'research_input'='materials/research_input'; 'reports'='nir/results'
}
# Resolve and validate every absolute source/destination before any move.
$layoutMoves = @()
foreach ($layoutEntry in $layoutMap.GetEnumerator()) {
    $layoutSource = [IO.Path]::GetFullPath((Join-Path $layoutRoot $layoutEntry.Key))
    $layoutDestination = [IO.Path]::GetFullPath((Join-Path $layoutRoot $layoutEntry.Value))
    if (-not $layoutSource.StartsWith($layoutRoot+'\',[StringComparison]::OrdinalIgnoreCase) -or -not $layoutDestination.StartsWith($layoutRoot+'\',[StringComparison]::OrdinalIgnoreCase)) { throw 'Move escapes intended workspace' }
    if (-not (Test-Path -LiteralPath $layoutSource)) { continue }
    if (Test-Path -LiteralPath $layoutDestination) { throw "Destination already exists: $layoutDestination" }
    $layoutMoves += [pscustomobject]@{Source=$layoutSource;Destination=$layoutDestination}
}
foreach ($layoutMove in $layoutMoves) {
    $layoutParent = Split-Path -Parent $layoutMove.Destination
    if (-not (Test-Path -LiteralPath $layoutParent)) { New-Item -ItemType Directory -Path $layoutParent | Out-Null }
    Move-Item -LiteralPath $layoutMove.Source -Destination $layoutMove.Destination
    Write-Output ($layoutMove.Source+' -> '+$layoutMove.Destination)
}
foreach ($layoutFolder in @('code','materials','nir','vkr')) {
    $layoutPath=Join-Path $layoutRoot $layoutFolder
    if (-not (Test-Path -LiteralPath $layoutPath)) { New-Item -ItemType Directory -Path $layoutPath | Out-Null }
}
