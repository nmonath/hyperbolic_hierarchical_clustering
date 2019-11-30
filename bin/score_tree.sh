#!/usr/bin/env bash

set -exu

tree=$1
algorithm=${2:-alg}
dataset=${3:-dataset}
threads=${4:-24}
expected_dp_point_file=${5:-"None"}

java -Xmx50G -cp target/ghhc-0.1-SNAPSHOT-jar-with-dependencies.jar ghhc.eval.EvalDendrogramPurity \
--input $tree --algorithm $algorithm --dataset $dataset --threads $threads \
--print true --id-file $expected_dp_point_file