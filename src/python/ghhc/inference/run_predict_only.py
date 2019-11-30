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

import sys
import os
import datetime
import numpy as np

from absl import logging
from absl import flags
from absl import app

import tensorflow as tf

from ghhc.util.Config import Config
from ghhc.util.load import load
from ghhc.util.initializers import random_pts
from ghhc.model.ghhc import gHHCTree, gHHCInference
from ghhc.util.io import mkdir_p

tf.enable_eager_execution()

FLAGS = flags.FLAGS

flags.DEFINE_string('config', None, "Config file")
flags.DEFINE_string('data_filename',None, 'data filename')
flags.DEFINE_string('output_filename', 'tree.tsv', 'output filename')
logging.set_verbosity(logging.INFO)


def main(argv):
  config = Config(FLAGS.config)

  filename = FLAGS.data_filename if FLAGS.data_filename is not None else config.inference_file

  assert(config.exp_out_dir is not None)
  assert (config.last_model is not None)

  config.exp_out_dir = os.path.join(config.exp_out_dir, 'results')
  mkdir_p(config.exp_out_dir)
  config.save_config(config.exp_out_dir,config.to_filename() + ".json")
  config.save_config(config.exp_out_dir)

  pids, lbls, dataset = load(filename, config)

  dev_pids, dev_lbls, dev_dataset = load(config.dev_file, config)

  init_tree = random_pts(dataset, config.num_internals, config.random_pts_scale)

  tree = gHHCTree(init_tree.copy(), config=config)
  optimizer = tf.train.GradientDescentOptimizer(config.tree_learning_rate)
  inf = gHHCInference(tree, optimizer, config, dev_dataset, dev_lbls)
  inf.ckpt.restore(tf.train.latest_checkpoint(inf.config.checkpoint_dir))

  tree.write_tsv(config.exp_out_dir + "/" + FLAGS.output_filename, dataset, lbls=lbls, pids=pids)

if __name__ == '__main__':
  app.run(main)