<#
PowerShell script to set a custom pagefile size on Windows.
Run as Administrator.
This will set the pagefile on C: to initial size 16384 MB and maximum 32768 MB (16-32 GB).
#>

$drive = "C:"
$initialMB = 16384
$maximumMB = 32768

Write-Host "Setting pagefile on $drive to $initialMB MB initial, $maximumMB MB maximum (requires admin)"

# Disable automatic pagefile management
wmic computersystem where name="%computername%" set AutomaticManagedPagefile=False

# Set the pagefile
wmic pagefileset where name="$drive\\pagefile.sys" delete
wmic pagefileset create name="$drive\\pagefile.sys" InitialSize=$initialMB,MaximumSize=$maximumMB

Write-Host "Pagefile updated. A reboot is required for changes to take effect."