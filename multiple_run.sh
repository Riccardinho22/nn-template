#!/bin/bash
set -e
NUM_CLIENTS=5

echo "Starting $NUM_CLIENTS clients."
for ((i = 0; i < $NUM_CLIENTS; i++))
do
    echo "Starting client(cid=$i) with partition $i out of $NUM_CLIENTS clients."
    PYTHONPATH=. python3 ./src/client_run.py -m client.cid=$i
done
echo "Started $NUM_CLIENTS clients."
