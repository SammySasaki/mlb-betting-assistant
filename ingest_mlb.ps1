# Set working directory
Set-Location "C:\Users\sasak\sports-bet"

# Log start time
$date = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
Add-Content -Path "logs\results_ingest.log" -Value "`n----- $date - Starting ingestion -----"

# Helper function to run a script and log output
function Run-IngestJob($scriptName) {
    $processInfo = New-Object System.Diagnostics.ProcessStartInfo
    $processInfo.FileName = "docker"
    $processInfo.Arguments = "compose run --rm app python app/ingestion/$scriptName"
    $processInfo.WorkingDirectory = "C:\Users\sasak\sports-bet"
    $processInfo.RedirectStandardOutput = $true
    $processInfo.RedirectStandardError = $true
    $processInfo.UseShellExecute = $false
    $processInfo.CreateNoWindow = $true

    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $processInfo
    $process.Start() | Out-Null

    $stdout = $process.StandardOutput.ReadToEnd()
    $stderr = $process.StandardError.ReadToEnd()

    $process.WaitForExit()
    $exitCode = $process.ExitCode

    Add-Content -Path "logs\results_ingest.log" -Value "`n--- Output from $scriptName ---"
    Add-Content -Path "logs\results_ingest.log" -Value $stdout
    Add-Content -Path "logs\results_ingest.log" -Value $stderr

    $date = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    if ($exitCode -eq 0) {
        Add-Content -Path "logs\results_ingest.log" -Value "$date - $scriptName completed successfully."
    } else {
        Add-Content -Path "logs\results_ingest.log" -Value "$date - ERROR: $scriptName failed with exit code $exitCode"
    }
}

# Run each script in order
Run-IngestJob "ingest_upcoming.py"
Run-IngestJob "ingest_odds.py"
Run-IngestJob "backfill_mlb_games.py yesterday"
Run-IngestJob "run_predictions.py"