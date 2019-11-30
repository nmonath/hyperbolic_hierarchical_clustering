"""
Copyright (C) 2019 Authors of gHHC
This file is part of "hyperbolic_hierarchical_clustering"
http://github.com/nmonath/hyperbolic_hierarchical_clustering
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os


def eval_dp(filename, outfile, threads, points_file, model_name='ghhc', dataset_name='dataset'):
    """Evaluate dendrogram purity with shell script using xcluster DP code."""

    os.system("sh bin/score_tree.sh {} {} {} {} {} > {}"
              .format(filename, model_name, dataset_name, threads, points_file, outfile))
    cost = None
    with open(outfile, 'r') as fin:
        for line in fin:
            splt = line.strip().split("\t")
            cost = float(splt[-1])
    return cost
