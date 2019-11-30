#!/usr/bin/env bash

set -exu

input=$1
filename=$2
out_file=${3:-"tree.tsv"}


python -m ghhc.inference.run_predict_only --config $input --data_filename $filename --output_filename $out_file