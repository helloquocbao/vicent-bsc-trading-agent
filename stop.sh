#!/bin/zsh
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

if [ -f logs/agent.pid ]; then
  kill $(cat logs/agent.pid) 2>/dev/null && echo "Agent stopped" || echo "Agent not running"
  rm logs/agent.pid
else
  echo "No agent PID file found"
fi

if [ -f logs/dashboard.pid ]; then
  kill $(cat logs/dashboard.pid) 2>/dev/null && echo "Dashboard stopped" || echo "Dashboard not running"
  rm logs/dashboard.pid
else
  echo "No dashboard PID file found"
fi
