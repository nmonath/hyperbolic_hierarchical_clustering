#!/usr/bin/env bash

set -exu

config=$1

python -m ghhc.util.sample_triples $config