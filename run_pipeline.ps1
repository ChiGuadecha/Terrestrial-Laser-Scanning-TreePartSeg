# ------------------------------------------------------------------
# 梁艺萌
# 树干 / 树枝 / 树叶 三维点云分离识别 —— 一键流水线
# 未经本人允许不得商用。
# ------------------------------------------------------------------
# 用法：
#   .\run_pipeline.ps1 -Stage prep
#   .\run_pipeline.ps1 -Stage train
#   .\run_pipeline.ps1 -Stage val  -Ids all
#   .\run_pipeline.ps1 -Stage test -Ids all
#   .\run_pipeline.ps1 -Stage deliver
#   .\run_pipeline.ps1 -Stage report
#   .\run_pipeline.ps1 -Stage all -Root D:\work\trees -Ptv3Local
param(
    [ValidateSet("relocate","prep","train","val","test","tune","deliver","report","all")]
    [string]$Stage = "all",
    [string]$Root = "",
    [switch]$Ptv3Local,
    [string]$Ids = "all",
    [string]$Run3 = "ptv3_task3_v4",
    [string]$Run2 = "ptv3_task2_v3",
    [string]$Grids = "0.02,0.015",
    [int]$Epochs = 120
)
$ErrorActionPreference = "Stop"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:GRIDS = $Grids
$PKG = $PSScriptRoot
$SRC = Join-Path $PKG "src"

function Invoke-Py([string[]]$Args_) {
    Write-Host ("> python " + ($Args_ -join " ")) -ForegroundColor Cyan
    Push-Location $SRC
    try { & python @Args_; if ($LASTEXITCODE -ne 0) { throw "python exit $LASTEXITCODE" } }
    finally { Pop-Location }
}

$want = { param($name) $Stage -eq "all" -or $Stage -eq $name }

if (& $want "relocate" -or $Root) {
    $a = @("relocate.py"); if ($Root) { $a += $Root }; if ($Ptv3Local) { $a += "--ptv3-local" }
    if ($Root) { Invoke-Py $a }
}
if (& $want "prep")   { Invoke-Py @("data_prep.py") }
if (& $want "train")  {
    Invoke-Py @("dl_sonata.py")
    Invoke-Py @("train.py", "--task", "3", "--name", $Run3, "--epochs", "$Epochs", "--oversample", "0.6", "--lw-branch", "2.0")
    Invoke-Py @("train.py", "--task", "2", "--name", $Run2, "--epochs", "$Epochs")
}
if (& $want "val")    { Invoke-Py @("predict.py", "val", $Run3, $Run2, $Ids) }
if (& $want "test")   { Invoke-Py @("predict.py", "test", $Run3, $Run2, $Ids) }
if (& $want "tune")   { Invoke-Py @("ablate_cache.py", "v4", $Run3); Invoke-Py @("sweep.py", "..\configs\cfg4.json") }
if (& $want "deliver"){
    Invoke-Py @("make_deliverables.py", "val")
    Invoke-Py @("make_deliverables.py", "test")
    Invoke-Py @("make_summary.py")
    Invoke-Py @("make_viewer.py", "pred_test_*.npz", "25000", "三维查看器_测试集.html")
    Invoke-Py @("make_viewer.py", "pred_val_*.npz", "25000", "三维查看器_验证集.html")
}
if (& $want "report") { Invoke-Py @("make_workflow_png.py") }
Write-Host "done: $Stage" -ForegroundColor Green
