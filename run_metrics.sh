#!/bin/bash
#SBATCH --mem=32G

# interactive run script with nohup
# runs the python analysis in background and logs to analysis.log

echo "================================================"
echo "Starting ERA5 vs CONUS404 Comparison Analysis"
echo "================================================"
echo ""
echo "Use 'tail -f out.log' to watch progress"
echo ""

# run python script in background with nohup
rm analysis.log
nohup python3 metrics.py > /dev/null 2>&1 &

# get process id
PID=$!

echo "Process started with PID: $PID"
echo "Log file: out.log"
echo ""
echo "To monitor: tail -f out.log"
echo "To check status: ps -p $PID"
echo "To stop: kill $PID"
echo ""
