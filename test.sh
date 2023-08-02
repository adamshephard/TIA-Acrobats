#!/usr/bin/env bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

echo "Building docker"
./build.sh

echo "Removing volume..."
docker volume rm tia-acrobats

echo "Creating volume..."
docker volume create tia-acrobats
# echo $SCRIPTPATH/testinput/
echo "Running algorithm..."

docker run --rm \
        --memory=$MEMORY \
        --memory-swap=$MEMORY \
        --network=none \
        --cap-drop=ALL \
        --security-opt="no-new-privileges" \
        --shm-size=128m \
        --pids-limit=256 \
        -v tia-acrobats:/output/ \
        --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 \
        tia-acrobats

