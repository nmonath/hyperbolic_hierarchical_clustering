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

import numpy as np

import tensorflow as tf

from absl import logging

from ghhc.model.ghhc import squared_euclidean_cdist
from sklearn.cluster import AgglomerativeClustering

tf.enable_eager_execution()


def euc_dist_batched(x_i, nodes, batch_size=1000):
    """Batched cdist operation."""
    dists = np.zeros((x_i.shape[0], nodes.shape[0]), np.float32)
    for i in range(0, x_i.shape[0], batch_size):
        logging.log_every_n_seconds(logging.INFO, 'euc_dist_batched processed %s of %s', 5, i, x_i.shape[0])
        for j in range(0, nodes.shape[0], batch_size):
            logging.log_every_n_seconds(logging.INFO, 'euc_dist_batched processed %s of %s', 5, j, nodes.shape[0])
            dists[i:(i + batch_size), j:(j + batch_size)] = squared_euclidean_cdist(
                x_i[i:(i + batch_size), :], nodes[j:(j + batch_size), :]).numpy()
    return dists


def afkmc2(data, k, m=20):
    """Implementation of Fast and Provably Good Seedings for k-Means https://las.inf.ethz.ch/files/bachem16fast.pdf """
    n = data.shape[0]
    c1 = np.random.randint(data.shape[0])
    c1_vec = np.expand_dims(data[c1], 0)
    q_nom = np.squeeze(euc_dist_batched(c1_vec, data))
    q_denom = np.sum(q_nom)
    q = 0.5 * q_nom / q_denom + 1.0 / (2.0 * n)
    indices = np.arange(n)
    c_i_minus_1 = np.zeros((k, data.shape[1]), dtype=np.float32)
    c_i_minus_1[0, :] = c1_vec
    for i in range(1, k):
        logging.log_every_n_seconds(logging.INFO, 'afkmc2 processed %s of %s', 5, i, k)
        x_ind = np.random.choice(indices, p=q)
        x = np.expand_dims(data[x_ind], 0)
        d_x = np.min(np.squeeze(euc_dist_batched(x, c_i_minus_1[:i])))
        for j in range(1, m):
            y_ind = np.random.choice(indices, p=q)
            y = np.expand_dims(data[y_ind], 0)
            d_y = np.min(np.squeeze(euc_dist_batched(y, c_i_minus_1[:i])))
            if ((d_y * q[x_ind]) / (d_x * q[y_ind])) > np.random.rand():
                x = y
                d_x = d_y
        c_i_minus_1[i] = x
    return c_i_minus_1


def init_from_rand_and_hac(data, k, scale):
    """Pick random points for leaves, find internals with HAC heuristic."""
    centers = random_pts(data, int(k / 2), 1.0)
    hac_pts = init_from_hac(centers, centers.shape[0] - 1)
    res = np.zeros((k, data.shape[1]), dtype=np.float32)
    assert k % 2 == 0
    res[0] += scale * data[np.random.randint(data.shape[0])]
    res[1:centers.shape[0] + 1, :] = centers
    res[centers.shape[0] + 1:, :] = hac_pts
    res = tf.clip_by_norm(scale * res, 0.80, axes=[1]).numpy()
    return res


def init_from_afkmc2_and_hac(data, k):
    """Pick leaves using afkmc2, find internals with HAC heuristic"""
    centers = afkmc2(data, int(k / 2))
    hac_pts = init_from_hac(centers, centers.shape[0] - 1)
    res = np.zeros((k, data.shape[1]), dtype=np.float32)
    assert k % 2 == 0
    res[0] += 0.65 * data[np.random.randint(data.shape[0])]
    res[1:centers.shape[0] + 1, :] = centers
    res[centers.shape[0] + 1:, :] = hac_pts
    res = tf.clip_by_norm(0.65 * res, 0.80, axes=[1]).numpy()
    return res


def hac_scaling_factor(n):
    return np.log2(n + 1 - np.arange(n)) / np.log2(n + 1)


def init_from_hac(data, k):
    """Find internal structure using hac heuristic"""
    agg = AgglomerativeClustering(n_clusters=1, linkage='average')
    agg.fit(data)
    internals = np.zeros((data.shape[0] - 1, data.shape[1]), dtype=np.float32)
    counts = np.zeros((data.shape[0] - 1), dtype=np.float32)
    children = agg.children_

    # find each agglomeration vector and
    def get_vector_for_idx(idx):
        if idx < data.shape[0]:
            return data[idx]
        else:
            return internals[idx - data.shape[0]]

    def get_count_for_idx(idx):
        if idx < data.shape[0]:
            return 1
        else:
            return counts[idx - data.shape[0]]

    for i in range(0, children.shape[0]):
        internals[i, :] = get_vector_for_idx(children[i, 0]) + get_vector_for_idx(children[i, 1])
        counts[i] = get_count_for_idx(children[i, 0]) + get_count_for_idx(children[i, 1])

    mean_internals = internals / np.expand_dims(counts, 1)
    normalized_internals = mean_internals / np.linalg.norm(mean_internals, axis=1, keepdims=True)
    selected_internals = normalized_internals[-k:, :]
    # print(mean_internals.shape)
    # print(normalized_internals.shape)
    # print(selected_internals.shape)
    # print(k)
    sf = hac_scaling_factor(data.shape[0])[-k:]
    # print(sf.shape)
    result = selected_internals * np.expand_dims(sf, 1)
    return result


def random_pts(data, n, scale):
    """Pick random points"""
    x_sample = np.random.choice(data.shape[0], size=n, replace=False)
    return scale * data[x_sample, :]


def random(data, n, scale):
    """Sample random points from normal(0,1)"""
    sample = np.random.randn(n, data.shape[1]).astype(np.float32)
    return scale * sample
