#!/bin/zsh
# Automatically cd to the directory containing this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# Default configuration
MODE="paper"  # Default is paper (demo) mode
PORT="9090"   # Default dashboard port

# Read command line arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    -m|--mode)
      MODE="$2"
      shift
      shift
      ;;
    -p|--port)
      PORT="$2"
      shift
      shift
      ;;
    *)
      echo "Invalid argument: $1. Usage: ./start.sh [--mode paper|live] [--port PORT_NUMBER]"
      exit 1
      ;;
  esac
done

# Create logs directory if it doesn't exist
mkdir -p logs

# Kill existing processes using PID files if they exist to avoid generic pkill conflicts
if [ -f logs/agent.pid ]; then
  kill $(cat logs/agent.pid) 2>/dev/null || true
  rm logs/agent.pid
fi
if [ -f logs/dashboard.pid ]; then
  kill $(cat logs/dashboard.pid) 2>/dev/null || true
  rm logs/dashboard.pid
fi
sleep 1

# Start agent in background using python3 -m
PYTHONPATH=src nohup python3 -m vicent.cli run --mode "$MODE" --interval 300 > logs/agent.log 2>&1 &
echo $! > logs/agent.pid
echo "Agent PID: $! (Mode: $MODE)"

# Start dashboard in background using python3 -m  
PYTHONPATH=src nohup python3 -m vicent.cli serve --port "$PORT" > logs/dashboard.log 2>&1 &
echo $! > logs/dashboard.pid
echo "Dashboard PID: $!"

echo "--------------------------------------------------------"
echo "VICENT Agent & Dashboard successfully started."
echo "Active mode: $MODE (demo = paper, live = active trading)"
echo "View Agent logs: tail -f logs/agent.log"
echo "View Dashboard logs: tail -f logs/dashboard.log"
echo "Dashboard URL: http://127.0.0.1:$PORT"
echo "--------------------------------------------------------"
