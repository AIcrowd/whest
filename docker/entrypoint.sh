#!/bin/sh
# Wait for the server socket to appear before running the submission
echo "Waiting for mechestim server..."
for i in $(seq 1 30); do
    if [ -e /tmp/mechestim.sock ]; then
        echo "Server socket found. Running submission."
        exec python /submission/run.py
    fi
    sleep 0.2
done
echo "ERROR: mechestim server socket not found after 6 seconds"
exit 1
