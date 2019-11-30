#!/usr/bin/env bash

set -exu

config=$1
filename=$2
partition=$3
mem=${4:-12000}
out_file=${5:-"tree.tsv"}

EMAIL=None
TIME=`(date +%Y-%m-%d-%H-%M-%S)`

threads=`cat $config | jq .threads`

export MKL_NUM_THREADS=$threads
export OPENBLAS_NUM_THREADS=$threads
export OMP_NUM_THREADS=$threads

log_dir=logs/inf/$TIME

mkdir -p $log_dir


sbatch -J inf-po-$TIME \
            -e $log_dir/inf.err \
            -o $log_dir/inf.log \
            --cpus-per-task $threads \
            --partition=$partition \
            --ntasks=1 \
            --nodes=1 \
            --mem=$mem \
            --time=0-08:00 \
            bin/run_predict_only.sh $config $filename $out_file