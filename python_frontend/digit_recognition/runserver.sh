host='132.145.139.222'
port='7050'

# KIll first
PIDS=$(pgrep -f "python3 manage.py")

# If the PID is not empty, kill the process
if [ -n "$PIDS" ]; then
    for PID in $PIDS; do
        kill -9 "$PID"
        echo "Process with PID $PID killed successfully."
    done
fi

# Start after
exec python3 manage.py runserver "0.0.0.0:$port" &> /dev/null &
echo "Running server in"
echo PID is $(pgrep -f "python3 manage.py")
echo Access GUI in    "$host:$port"
