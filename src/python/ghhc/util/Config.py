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

import json
import random
import os
import numpy as np


class Config(object):
    """Config object"""

    def __init__(self,filename=None):
        # Settings
        self.config_name = filename
        self.dataset_name = 'dataset'
        self.alg_name = 'ghhc'
        self.exp_out_base = 'exp_out'
        self.exp_out_dir = None

        self.inference_file = None

        # Preprocessing
        self.unit_norm = True
        self.zero_mean = True

        # ghhc
        self.max_norm = 0.9
        self.init_method = 'randompts'
        self.batch_size = 500
        self.tree_learning_rate = 0.01
        self.num_internals = 100
        self.random_pts_scale = 0.001
        self.num_iterations = 10
        self.shuffle = True
        self.dev_points_file = 'None'
        self.gamma = 1.0
        self.struct_prior_every = 100
        self.num_struct_prior_batches = 100
        self.struct_prior = 'pcn'
        self.last_model = None

        # Triplet Sampling
        self.num_samples = 50000
        self.percent_random = 0.1
        self.sample_file = None
        self.sample_dataset = None
        self.triplet_k = 5
        self.nn_batch_size = 1000
        self.sample_outfile = None

        self.dev_file = None
        self.dev_every = 10000
        self.save_dev_pics = False
        self.loss_type = 'threespread'
        self.loss = 'sigmoid'
        self.lca_type = 'conditional'
        self.threads = 1
        self.checkpoint_dir = None
        self.save_every = 100000
        self.random_projection = None
        self.random_seed = 1451

        self.episode_size = 5000

        np.random.seed(self.random_seed)

        if filename:
            self.__dict__.update(json.load(open(filename)))
        self.random = random.Random(self.random_seed)

    def to_json(self):
        return json.dumps(self.filter_json(self.__dict__),indent=4,sort_keys=True)

    def save_config(self, exp_dir, filename='config.json'):
        with open(os.path.join(exp_dir, filename), 'w') as fout:
            fout.write(self.to_json())
            fout.write('\n')

    def to_file_name_from_fields(self, fields):
        return "-".join(["%s=%s" % (f,self.__dict__[f]) for f in fields])

    def to_filename(self):
        return self.to_file_name_from_fields(['alg_name', 'init_method', 'tree_learning_rate',
                                              'loss', 'lca_type', 'num_samples',
                                              'batch_size', 'struct_prior'])

    def filter_json(self, the_dict):
        # print("filter_json")
        # print(the_dict)
        res = {}
        for k in the_dict.keys():
            # print("k : {} \t {} \t {}".format(k,the_dict[k],type(the_dict[k])))
            if type(the_dict[k]) is str or \
                type(the_dict[k]) is float or \
                type(the_dict[k]) is int or \
                type(the_dict[k]) is list or \
                type(the_dict[k]) is bool or \
                the_dict[k] is None:
                # print("res[k] : {} \t {} \t {}".format(k, the_dict[k], type(the_dict[k])))
                res[k] = the_dict[k]
            elif type(the_dict[k]) is dict:
                res[k] = self.filter_json(the_dict[k])
        return res

DefaultConfig = Config()


