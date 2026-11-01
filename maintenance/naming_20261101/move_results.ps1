$ErrorActionPreference='Stop'
$root='C:\Users\lenovo\Desktop\eVTOL_Simulation'
$pairs=@(
    @('simulink\output\recovery','simulink\output\simulations'),
    @('simulink\output\simulations\recovery_v3_20261023','simulink\output\simulations\control_v3_20261023'),
    @('simulink\output\recovery_loops','simulink\output\loop_tests'),
    @('simulink\output\recovery_validation','simulink\output\validation'),
    @('simulink\output\validation\recovery_v3_20261023','simulink\output\validation\control_v3_20261023')
)
foreach($pair in $pairs){
    $source=(Resolve-Path -LiteralPath (Join-Path $root $pair[0])).ProviderPath
    $destination=[IO.Path]::GetFullPath((Join-Path $root $pair[1]))
    if(-not $source.StartsWith($root+'\',[StringComparison]::OrdinalIgnoreCase) -or -not $destination.StartsWith($root+'\',[StringComparison]::OrdinalIgnoreCase)){throw 'Rename escaped workspace.'}
    if(Test-Path -LiteralPath $destination){throw "Destination exists: $destination"}
    Move-Item -LiteralPath $source -Destination $destination
}
'Result folders renamed within the workspace.'
