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

echo "Starting ROS nodes with a clean roscore..."
exec roslaunch startup_scripts start_final_all.launch
