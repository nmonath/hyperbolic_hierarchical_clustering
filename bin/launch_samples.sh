#!/usr/bin/env bash

set -exu

config=$1
partition=$2
mem=${3:-12000}

threads=`cat $config | jq .threads`

EMAIL=None
TIME=`(date +%Y-%m-%d-%H-%M-%S)`

export MKL_NUM_THREADS=$threads
export OPENBLAS_NUM_THREADS=$threads
export OMP_NUM_THREADS=$threads

log_dir=logs/samples/$TIME

mkdir -p $log_dir


sbatch -J samples-$TIME \
            -e $log_dir/samples.err \
            -o $log_dir/samples.log \
            --cpus-per-task $threads \
            --partition=$partition \
            --ntasks=1 \
            --nodes=1 \
            --mem=$mem \
            --time=0-10:00 \
            bin/sample_triples.sh $config