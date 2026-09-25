# PowerShell convenience runner for all prerequisites and pipeline
[CmdletBinding()]
param(
    [string]$Mode = "both",
    [switch]$Synthetic,
    [switch]$SkipInstall,
    [switch]$SkipTests
)

$argsList = @("run_all.py", "--mode", $Mode)
if ($Synthetic) { $argsList += "--synthetic" }
if ($SkipInstall) { $argsList += "--skip-install" }
if ($SkipTests) { $argsList += "--skip-tests" }

python @argsList
