while($true) {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $gpu_stats = nvidia-smi --query-gpu=temperature.gpu,utilization.gpu --format=csv,noheader
    $log_entry = "$timestamp, $gpu_stats"
    Out-File -Append -FilePath gpu_thermal_log.csv -InputObject $log_entry
    Start-Sleep -Seconds 120
}
