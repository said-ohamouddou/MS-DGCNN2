#!/bin/bash
# Call nvidia-smi to get the list of processes using the GPU
# Filter the output to get only Python processes and their PIDs
# Remove any trailing commas from the PIDs
# Kill each Python process

echo "Checking GPU processes..."

# Extract PIDs of Python processes using the GPU, remove commas at the end
pids=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | grep python | awk '{print $1}' | sed 's/,//')

# Loop through the PIDs and kill each one
for pid in $pids; do
    echo "Killing Python process with PID: $pid"
    kill -9 $pid
done

echo "All Python GPU processes have been killed."

