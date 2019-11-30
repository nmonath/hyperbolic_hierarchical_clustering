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

from sklearn.neighbors import NearestNeighbors

import sys
import numpy as np
from absl import logging
from ghhc.util.Config import Config
from ghhc.util.load import load

logging.set_verbosity(logging.INFO)

def batch_find_neighbors(X,nn: NearestNeighbors,batch_size=1000):
  res = np.zeros((X.shape[0], nn.n_neighbors-1), dtype=np.int32)
  for i in range(0, X.shape[0], batch_size):
    logging.log_every_n_seconds(logging.INFO, 'Computed %s of %s neighbors' % (i,X.shape[0]), 5)
    res[i:(i + batch_size)] = nn.kneighbors(X[i:(i+batch_size), :], return_distance=False)[:,1:]
  logging.info('Finished batch_find_neighbors')
  return res

def order_triple(X, i, j, k, comparison):
  ij = comparison(X[i, :], X[j, :])
  jk = comparison(X[j, :], X[k, :])
  ik = comparison(X[i, :], X[k, :])
  if ij >= max(jk, ik):
    return np.array([i, j, k], dtype=np.int32)
  elif jk >= max(ij,ik):
    return np.array([j, k, i], dtype=np.int32)
  elif ik >= max(ij, jk):
    return np.array([i, k, j], dtype=np.int32)

def sample_random(X, comparison):
  ijk = np.random.choice(X.shape[0],size=3,replace=False)
  return order_triple(X, ijk[0], ijk[1], ijk[2], comparison)

def sample_nn(X, Xnn, comparison):
  ik = np.random.choice(Xnn.shape[0],size=2,replace=False)
  i = ik[0]
  k = ik[1]
  j = np.random.choice(Xnn[i,:])
  while k == i or k == j:
    k = np.random.randint(X.shape[0],size=1)[0]
  return order_triple(X, i, j, k, comparison)

if __name__ == "__main__":
  config = Config(sys.argv[1])
  pids, lbls, X = load(config.sample_dataset, config)
  nn = NearestNeighbors(n_neighbors=config.triplet_k+1, algorithm='ball_tree').fit(X)
  Xnn = batch_find_neighbors(X, nn, config.nn_batch_size)
  samples = np.zeros((config.num_samples, 3), dtype=np.int32)
  def sim_fn(i, j):
    return 1.0 / (1.0 + np.linalg.norm(i - j))
  for i in range(config.num_samples):
    if np.random.random() < config.percent_random:
      samples[i, :] = sample_random(X, sim_fn)
    else:
      samples[i, :] = sample_nn(X, Xnn, sim_fn)
  np.save(config.sample_outfile, samples)


