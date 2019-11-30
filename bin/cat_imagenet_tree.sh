#!/usr/bin/env bash

set -exu

results_dir=$1

pushd $results_dir
cat tree-aa.tsv.leaves tree-ab.tsv.leaves tree-ac.tsv.leaves tree-ad.tsv.leaves tree-ae.tsv.leaves tree-af.tsv.leaves tree-ag.tsv.leaves tree-ah.tsv.leaves tree-ai.tsv.leaves tree-aj.tsv.leaves tree-ak.tsv.leaves tree-al.tsv.leaves tree-am.tsv.leaves tree-aa.tsv.internals > tree.tsv
popd