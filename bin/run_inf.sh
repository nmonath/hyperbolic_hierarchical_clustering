#!/usr/bin/env bash

set -exu

input=$1

python -m ghhc.inference.run_inference $input