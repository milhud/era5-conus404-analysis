#!/bin/bash

# interactive run script with nohup
# runs the python analysis in background and logs to analysis.log

echo "================================================"
echo "Starting ERA5 vs CONUS404 Comparison Analysis"
echo "================================================"
echo ""
echo "Use 'tail -f analysis.log' to watch progress"
echo ""

# run python script in background with nohup
rm analysis.log
nohup python3 analysis_script.py > /dev/null 2>&1 &

# get process id
PID=$!

echo "Process started with PID: $PID"
echo "Log file: analysis.log"
echo ""
echo "To monitor: tail -f analysis.log"
echo "To check status: ps -p $PID"
echo "To stop: kill $PID"
echo ""
