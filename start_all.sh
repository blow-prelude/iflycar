#!/usr/bin/env bash

set -e

source /home/ucar/ucar_ws/devel/setup.bash

echo "Stopping the old roscore..."
sudo pkill roscore || true

old_master_stopped=false
for _ in {1..50}; do
  if ! rosparam list >/dev/null 2>&1; then
    old_master_stopped=true
    break
  fi
  sleep 0.1
done

if [ "$old_master_stopped" != true ]; then
  echo "the old roscore did not stop" >&2
  exit 1
fi

echo "Starting a clean roscore..."
roscore &
roscore_pid=$!

cleanup() {
  kill "$roscore_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

master_ready=false
for _ in {1..50}; do
  if rosparam list >/dev/null 2>&1; then
    master_ready=true
    break
  fi
  sleep 0.1
done

if [ "$master_ready" != true ]; then
  echo "roscore did not become ready" >&2
  exit 1
fi

echo "Clearing all ROS parameters..."
rosparam delete /

echo "Waiting 2 seconds before starting nodes..."
sleep 2

roslaunch startup_scripts start_all.launch clear_params:=false
