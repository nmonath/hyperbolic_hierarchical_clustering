#!/usr/bin/env bash

set -exu

config=$1
filename=$2
partition=$3
mem=${4:-12000}

split_names=( "aa" "ab" "ac" "ad" "ae" "af" "ag" "ah" "ai" "aj" "ak" "al" "am" )
#split_names=( "aa" "ab" "ac" )
#split_names=( "ad" "ae" "af" "ag" "ah" "ai" "aj" "ak" "al" "am" )

for i in "${split_names[@]}"
do
	sh bin/launch_predict_only.sh $config "${filename}.split_${i}" $partition $mem "tree-$i.tsv"
	sleep 1
done