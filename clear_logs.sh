#!/bin/bash

if [ -z "$1" ]; then
    echo "No argument supplied"
else
    cmd="rm -rf models/$1/logs/"
    echo $cmd
    $cmd
fi
