#!/bin/bash

# Create a SIGINT handler
control_c() {
    # Terminate the camera screen
    python3 ~/start_nano.py -c
    # Exit the script
    exit
}
trap control_c SIGINT

# Get the TTY device for the current terminal
TTY=$(tty)

# This starts the camera screen
python3 ~/start_nano.py --tty=$TTY

# This idles until Ctrl+C is pressed. https://stackoverflow.com/a/36991974
read -r -d '' _ </dev/tty