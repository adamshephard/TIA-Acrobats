#!/usr/bin/env bash

./build.sh

docker save tia-acrobats | gzip -c > tia-acrobats-algorithm.tar.xz