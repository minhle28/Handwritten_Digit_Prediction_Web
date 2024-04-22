#!/bin/bash

# Find the PID of the process containing "python3 manay.py"
PIDS=$(pgrep -f "python3 manage.py")

# If the PID is not empty, kill the process
if [ -n "$PIDS" ]; then
    for PID in $PIDS; do
        kill -9 "$PID"
        echo "Process with PID $PID killed successfully."
    done
else
    echo "No matching processes found."
fi