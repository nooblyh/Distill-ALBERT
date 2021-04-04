#!/bin/sh

if [ -z "$1" ]
then
    yhalloc -N 1 -p gpu_v100 -J lyh-test
else
    yhalloc -N 1 -p "$1" -J lyh
fi
