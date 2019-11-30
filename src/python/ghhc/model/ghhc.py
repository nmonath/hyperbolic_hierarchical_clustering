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
import time

import numpy as np
import tensorflow as tf

from ghhc.util.eval_dp import eval_dp
from ghhc.util.io import mkdir_p

from absl import logging

tf.enable_eager_execution()

def squared_norm(x, axis=1, keepdims=True):
    """Squared L2 Norm of x."""
    return tf.reduce_sum(tf.pow(x, 2), axis=axis, keepdims=keepdims)


def squared_euclidean_cdist(x, y):
    """Squared euclidean distance

    Computed as: ||x||^2 + ||y||^2 - 2 x^T y.

    Args:
        x: N by D matrix
        y: M by D matrix
    :returns matrix (N by M) such that result[i,j] = || x[i,:] - y[j,;] ||^2
    """
    norms = squared_norm(x, axis=1, keepdims=True) + tf.transpose(squared_norm(y, axis=1, keepdims=True))
    dot = 2.0*tf.matmul(x, y, transpose_b=True)
    return norms - dot

def poincare_cdist(x, y):
    """Poincare distance

    Args:
        x: N by D matrix
        y: M by D matrix
    :returns matrix (N by M) such that result[i,j] = ppoincare dist(x[i,:], y[j,:])
    """
    numerator = squared_euclidean_cdist(x, y)
    denom = (1.0 - squared_norm(x)) * (1.0 - tf.transpose(squared_norm(y, axis=1, keepdims=True)))
    arccosh_arg = 1.0 + 2.0 * numerator / denom
    res = tf.math.acosh(1e-8 + arccosh_arg)
    return res


def squared_euclidean_dist(x, y):
    """Squared euclidean distance

    Computed as: ||x||^2 + ||y||^2 - 2 x^T y.

    Args:
        x: N by D matrix
        y: N by D matrix
    :returns vector (N by 1) such that the ith element is || x[i,:] - y[i,;] ||^2
    """
    norms = squared_norm(x, axis=1, keepdims=True) + squared_norm(y, axis=1, keepdims=True)
    dot = 2*tf.reduce_sum(tf.multiply(x, y), axis=1, keepdims=True)
    return norms - dot


def poincare_dist(x, y):
    """Poincare distance between x and y.

        Args:
            x: N by D matrix
            y: N by D matrix
        :returns vector (N by 1) such that the ith element is poincare dist(x[i,:], y[i,:])
        """
    numerator = squared_euclidean_dist(x, y)
    denom = (1.0 - squared_norm(x)) * (1.0 - squared_norm(y))
    arccosh_arg = 1.0 + 2.0 * numerator / denom
    res = tf.math.acosh(arccosh_arg)
    return res


def poincare_norm(x, axis=1, keepdims=True):
    """Squared poincare norm of x."""
    return 2.0*tf.math.atanh(tf.linalg.norm(x, axis=axis, keepdims=keepdims))


def parent_order_penalty(p, c, marg):
    """Penalty for parents to have smaller norm than children."""
    return tf.maximum(0.0, poincare_norm(p) - poincare_norm(c) + marg) + 1.0


def parent_order_penalty_cdist(p, c, marg):
    """Penalty for parents to have smaller norm than children."""
    return tf.maximum(0.0, tf.transpose(poincare_norm(p)) - poincare_norm(c) + marg) + 1.0


class gHHCTree(tf.keras.Model):
    """Object for a ghhc tree."""

    def __init__(self, init_tree=None, gamma=0.25, config=None, projection=None):
        super(gHHCTree, self).__init__()
        self.internals = tf.get_variable('internals', initializer=init_tree)
        self.max_norm = 0.8
        self.internals_so_far = 0
        self.gamma = gamma
        self.config = config
        self.projection = None
        self.cached_pairs = None
        if projection is not None:
            self.projection = projection
        else:
            self.projection = lambda x: x

    def project(self, x_i, x_j, x_k):
        return self.projection(x_i), self.projection(x_j), self.projection(x_k)

    def clip(self):
        tf.assign(self.internals, tf.clip_by_norm(self.internals, self.max_norm, axes=[1]))

    def p_par_broadcast(self, x_i):
        return self.p_par_to_broadcast(x_i, self.internals)

    def p_par_to_broadcast(self, x_i, nodes):
        dists = poincare_cdist(x_i, nodes)
        res = tf.multiply(dists, parent_order_penalty_cdist(nodes, x_i, self.gamma))
        return res

    def p_par_to(self, x_i, nodes):
        dists = poincare_dist(x_i, nodes)
        res = tf.multiply(dists, parent_order_penalty(nodes, x_i, self.gamma))
        return res

    def p_par_to_batched_np(self, x_i, nodes, batch_size=1000):
        dists = np.zeros((x_i.shape[0], nodes.shape[0]), np.float32)
        for i in range(0, x_i.shape[0], batch_size):
            logging.log_every_n_seconds(logging.INFO,'p_par_to_batched_np processed %s of %s', 5, i, x_i.shape[0])
            for j in range(0, nodes.shape[0], batch_size):
                dists[i:(i+batch_size), j:(j+batch_size)] = self.p_par_to_broadcast(x_i[i:(i + batch_size), :], nodes[j:(j + batch_size), :]).numpy()
        return dists

    def compute_loss(self, x_i, x_j, x_k):
        x_i, x_j, x_k = self.project(x_i, x_j, x_k)

        x_i_dists = self.p_par_to_broadcast(x_i, self.internals)
        x_j_dists = self.p_par_to_broadcast(x_j, self.internals)
        x_k_dists = self.p_par_to_broadcast(x_k, self.internals)

        max_dists_ij = tf.maximum(x_i_dists, x_j_dists)
        gumbel_ij_noise = tf.log(-tf.log(tf.random_uniform(tf.shape(max_dists_ij))))
        gumbel_ijk_noise = tf.log(-tf.log(tf.random_uniform(tf.shape(max_dists_ij))))
        max_dists_ijk = tf.maximum(x_k_dists, max_dists_ij)
        lca_ij_softmax = tf.nn.softmax(-max_dists_ij+gumbel_ij_noise, axis=1)
        lca_ij_idx = tf.argmin(max_dists_ij, axis=1)
        offset = np.zeros_like(max_dists_ij)
        offset[np.arange(offset.shape[0]), lca_ij_idx] = 1000

        max_dists_ijk += offset
        lca_ijk_softmax = tf.nn.softmax(-max_dists_ijk + gumbel_ijk_noise, axis=1)

        logits1 = lca_ij_softmax * x_i_dists - lca_ijk_softmax * x_i_dists
        logits2 = lca_ij_softmax * x_j_dists - lca_ijk_softmax * x_j_dists
        logits3 = lca_ijk_softmax * x_k_dists - lca_ij_softmax * x_k_dists

        per_ex_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(logits1), logits=logits1) \
                      + tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(logits2), logits=logits2) \
                      + tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(logits3), logits=logits3)
        loss = tf.reduce_sum(per_ex_loss)
        return loss

    def p_par_assign_to_internal(self, children, parents, proj_child=True):
        if proj_child:
            children = self.projection(children)

        internal_norm = tf.norm(parents, axis=1, keepdims=True)
        internal_ordering = tf.argsort(-tf.squeeze(internal_norm)).numpy()
        back_to_orig = tf.argsort(internal_ordering)
        parents = parents[internal_ordering,:]
        children = children[internal_ordering,:]

        dists = self.p_par_to_batched_np(children, parents)
        dists[np.tril_indices_from(dists)] = np.Inf
        np.fill_diagonal(dists, np.Inf)
        dists = dists[back_to_orig,:][:,back_to_orig]
        assignments = np.argmin(dists, axis=1)
        mindists = np.min(dists, axis=1)
        assignments[mindists == np.Inf] = -1
        return assignments

    def p_par_assign_to(self, children, parents, exclude_diag=False, proj_child=True):
        if proj_child:
            children = self.projection(children)
        dists = self.p_par_to_batched_np(children, parents)
        children_norm = tf.norm(children, axis=1, keepdims=True)
        internal_norm = tf.norm(parents, axis=1, keepdims=True)
        eligible = tf.less(-children_norm + tf.transpose(internal_norm), 0).numpy()
        dists[eligible == False] = np.Inf
        if exclude_diag:
            np.fill_diagonal(dists, np.Inf)
        assignments = np.argmin(dists, axis=1)
        mindists = np.min(dists, axis=1)
        assignments[mindists == np.Inf] = -1
        return assignments

    def write_tsv(self, filename, leaves, pids=None, lbls=None, update_cache=True):
        logging.info('Writing tree tsv to %s' % filename)
        logging.info('num leaves %s' % leaves.shape[0])
        logging.info('pids is None? %s' % (pids is None))
        logging.info('lbls is None? %s' % (lbls is None))
        internals = self.internals.numpy()
        leaf_to_par_assign = self.p_par_assign_to(leaves, internals)
        internal_to_par_assign = self.p_par_assign_to_internal(internals, internals, proj_child=False)
        self.cached_pairs = np.concatenate([ np.expand_dims(np.arange(internal_to_par_assign.shape[0]),1), np.expand_dims(internal_to_par_assign,1)],axis=1)
        self.cached_pairs = self.cached_pairs[self.cached_pairs[:,1]!=-1]
        with open(filename + '.internals', 'w') as fouti:
            with open(filename + '.leaves', 'w') as foutl:
                with open(filename, 'w') as fout:
                    i = -1
                    pid = 'int_%s' % i
                    best_pid = 'best_int_%s' % i
                    par_id = 'None'
                    fout.write('%s\t%s\tNone\n' % (pid, par_id))
                    fout.write('%s\t%s\tNone\n' % (best_pid, pid))

                    fouti.write('%s\t%s\tNone\n' % (pid, par_id))
                    fouti.write('%s\t%s\tNone\n' % (best_pid, pid))

                    for i in range(leaf_to_par_assign.shape[0]):
                        logging.log_every_n_seconds(logging.INFO,'Wrote %s leaves' % i,5)
                        pid = 'pt_%s' % i if pids is None else pids[i]
                        lbl = pid if lbls is None else lbls[i]
                        par_id = 'best_int_%s' % leaf_to_par_assign[i]
                        fout.write('%s\t%s\t%s\n' % (pid, par_id, lbl))
                        foutl.write('%s\t%s\t%s\n' % (pid, par_id, lbl))

                    for i in range(internal_to_par_assign.shape[0]):
                        logging.log_every_n_seconds(logging.INFO,'Wrote %s internals' % i,5)
                        pid = 'int_%s' % i
                        par_id = 'int_%s' % internal_to_par_assign[i]
                        best_pid = 'best_int_%s' % i
                        fout.write('%s\t%s\tNone\n' % (pid, par_id))
                        fout.write('%s\t%s\tNone\n' % (best_pid, par_id))
                        fouti.write('%s\t%s\tNone\n' % (pid, par_id))
                        fouti.write('%s\t%s\tNone\n' % (best_pid, par_id))

    def plot_tree(self, leaves, filename):
        internals = self.internals.numpy()
        leaf_to_par_assign = self.p_par_assign_to(leaves, internals)
        internal_to_par_assign = self.p_par_assign_to_internal(internals, internals, proj_child=False)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='x', which='both', bottom='off', top='off',
                                     color='white')
        ax.tick_params(axis='y', which='both', left='off', right='off',
                                     color='white')
        # plt.scatter(0, 0, label='root', marker='^', zorder=2)
        # plt.annotate('root', xy=(0,0), size=3)

        for idx in range(internals.shape[0]):
            plt.scatter(internals[idx, 0], internals[idx, 1], label='int_%s' % idx, s=100, marker='^', zorder=2)
            # plt.annotate('int_%s' % idx, xy=(internals[idx,0], internals[idx,1]), size=5)
        for idx in range(internals.shape[0]):
            if internal_to_par_assign[idx] != -1:
                plt.plot([internals[idx,0], internals[internal_to_par_assign[idx],0]],
                [internals[idx,1], internals[internal_to_par_assign[idx],1]], linewidth=2,
                                 c='k', zorder=1)
            # else:
                # plt.plot([internals[idx, 0], 0],
                #                    [internals[idx, 1], 0], linewidth=1,
                #                    c='k', zorder=1)
        for idx in range(leaves.shape[0]):
            plt.scatter(leaves[idx, 0], leaves[idx, 1], s=100, label='%s' % idx, marker='o', zorder=2)

            # plt.annotate('pt_%s' % idx, xy=(leaves[idx, 0], leaves[idx, 1]), size=5)
        for idx in range(leaves.shape[0]):
            if leaf_to_par_assign[idx] != -1:
                # print('gpid %s lpid %s' % (grinch_par_id, leaf_to_par_assign[idx]))
                plt.plot([leaves[idx, 0], internals[leaf_to_par_assign[idx], 0]],
                                 [leaves[idx, 1], internals[leaf_to_par_assign[idx], 1]], linewidth=2,
                                 c='k', zorder=1)
            # else:
            #     plt.plot([leaves[idx, 0], 0],
            #                        [leaves[idx, 1], 0], linewidth=1,
            #                        c='k', zorder=1)
        plt.xlim([-1.1, 1.1])
        plt.ylim([-1.1, 1.1])
        circle = plt.Circle((0, 0), 1, color='r',linewidth=5, fill=False)
        ax.add_artist(circle)
        plt.axis('off')
        plt.savefig(filename)

    def structure_loss(self):
        res = tf.reduce_sum(self.child_parent_norm_loss(self.cached_pairs))
        # logging.log_every_n(logging.INFO,'cp res: %s', 10,res )
        return res

    def child_parent_norm_loss(self, pairs):
        internal_norms = poincare_norm(self.internals)
        children = tf.gather(internal_norms, pairs[:,0])
        parents = tf.gather(internal_norms, pairs[:,1])
        logits1 = tf.nn.relu(parents - children + self.gamma)
        min_norm = tf.argmin(internal_norms).numpy()[0]
        logging.log_every_n(logging.INFO,'min_norm %s %s',500,min_norm,internal_norms[min_norm])
        max_norm = tf.argmax(internal_norms).numpy()[0]
        logging.log_every_n(logging.INFO, 'max_norm %s %s', 500, max_norm, internal_norms[max_norm])
        return tf.reduce_sum(logits1)


def rsgd_or_sgd(grads_and_vars, rsgd=True):
    if rsgd:
        res = []
        for g,v in grads_and_vars:
            scale = ((1.0 - tf.reduce_sum(tf.multiply(v,v),axis=1,keepdims=True)) ** 2) / 4.0
            res.append((scale*g, v))
        return res
    else:
        return grads_and_vars

class gHHCInference(object):
    def __init__(self, ghhcTree, optimizer, config, dev_set, dev_lbls):
        self.ghhcTree = ghhcTree
        self.optimizer = optimizer
        self.config = config
        self.dev_set = dev_set
        self.dev_lbls = dev_lbls
        self.best_dev_dp_score = 0.0
        self.best_dev_iter = 0.0
        self.last_dev_dp_score = 0.0
        self.last_dev_iter = 0.0
        self.checkpoint_prefix = self.config.checkpoint_dir + "/ckpt"
        self.ckpt = tf.train.Checkpoint(optimizer=optimizer,
                                                             model=ghhcTree,
                                                             optimizer_step=tf.train.get_or_create_global_step())

    def update(self, c1, c2, par_id, gp_id, steps=100):
        for i in range(steps):
            with tf.GradientTape() as tape:
                loss = self.ghhcTree.pull_close_par_gp(c1, c2, par_id, gp_id)
            grads = tape.gradient(loss, self.ghhcTree.trainable_variables)
            self.optimizer.apply_gradients(rsgd_or_sgd(zip(grads, self.ghhcTree.trainable_variables)),
                                                                         global_step=tf.train.get_or_create_global_step())
            self.ghhcTree.clip()

    def episode_inference(self, x_i, x_j, x_k, dataset, batch_size=1000, examples_so_far=0):
        time_so_far = 0.0
        loss_so_far = 0.0
        struct_loss_so_far = 0.0

        for idx in range(0, x_i.shape[0], batch_size):

            if self.config.struct_prior is not None and idx+examples_so_far > 0:
                if self.ghhcTree.cached_pairs is None:
                    self.dev_eval(idx + examples_so_far)

                if (idx + examples_so_far) % self.config.struct_prior_every == 0:
                    for idx2 in range(self.config.num_struct_prior_batches):
                        start_time = time.time()
                        logging.log_every_n(logging.INFO,
                            '[STRUCTURE] Processed %s of %s batches || Avg. Loss %s || Avg Time %s' % (idx2, 100, struct_loss_so_far / max(idx2, 1), time_so_far / max(idx2, 1)),100)
                        with tf.GradientTape() as tape:
                            sloss = self.ghhcTree.structure_loss()
                            struct_loss_so_far += sloss.numpy()
                        grads = tape.gradient(sloss, self.ghhcTree.trainable_variables)
                        self.optimizer.apply_gradients(rsgd_or_sgd(zip(grads, self.ghhcTree.trainable_variables)),
                                                                                     global_step=tf.train.get_or_create_global_step())
                        self.ghhcTree.clip()
                        end_time = time.time()
                        time_so_far += end_time - start_time
                    logging.log(logging.INFO, '[STRUCTURE] Processed %s of %s batches || Avg. Loss %s || Avg Time %s' % (self.config.num_struct_prior_batches, 100, struct_loss_so_far / max(self.config.num_struct_prior_batches, 1), time_so_far / max(self.config.num_struct_prior_batches, 1)))

            if (idx + examples_so_far) % self.config.dev_every == 0:
                self.dev_eval(idx + examples_so_far)
            elif (idx + examples_so_far ) % self.config.save_every == 0:
                self.ckpt.save(self.checkpoint_prefix)
                self.config.last_model = tf.train.latest_checkpoint(self.config.checkpoint_dir)
                self.config.save_config(self.config.exp_out_dir, filename='config.json')

            start_time = time.time()
            if idx % 100 == 0 and idx > 0:
                logging.info('Processed %s of %s batches || Avg. Loss %s || Avg Time %s' % (idx, x_i.shape[0], loss_so_far/idx, time_so_far / max(idx,1)))
            with tf.GradientTape() as tape:
                bx_i = dataset[x_i[idx:(idx + batch_size)], :]
                bx_j = dataset[x_j[idx:(idx + batch_size)], :]
                bx_k = dataset[x_k[idx:(idx + batch_size)], :]
                loss = self.ghhcTree.compute_loss(bx_i, bx_j, bx_k)
                loss_so_far += loss.numpy()
            grads = tape.gradient(loss, self.ghhcTree.trainable_variables)
            self.optimizer.apply_gradients(rsgd_or_sgd(zip(grads, self.ghhcTree.trainable_variables)),
                                                                global_step=tf.train.get_or_create_global_step())
            self.ghhcTree.clip()
            end_time = time.time()
            time_so_far += end_time - start_time

        logging.info('Processed %s of %s batches || Avg. Loss %s || Avg Time %s' % (x_i.shape[0], x_i.shape[0], loss_so_far / x_i.shape[0], time_so_far / max(x_i.shape[0], 1)))

        # save model at the end of training
        self.ckpt.save(self.checkpoint_prefix)
        self.config.last_model = tf.train.latest_checkpoint(self.config.checkpoint_dir)
        # record the last model in the config.
        self.config.save_config(self.config.exp_out_dir, filename='config.json')
        return x_i.shape[0]

    def dev_eval(self, steps):
        if self.dev_set is not None:
            start_dev = time.time()
            mkdir_p(os.path.join(self.config.exp_out_dir, 'dev'))
            filename = os.path.join(self.config.exp_out_dir, 'dev', 'dev_tree_%s.tsv' % steps)
            self.ghhcTree.write_tsv(filename,self.dev_set,lbls=self.dev_lbls)
            dp = eval_dp(filename, os.path.join(self.config.exp_out_dir, 'dev', 'dev_score_%s.tsv' % steps),
                                     self.config.threads, self.config.dev_points_file)
            logging.info('DEV EVAL @ %s minibatches || %s DP' % (steps,dp))
            end_dev = time.time()
            logging.info('Finished Dev Eval in %s seconds' % (end_dev-start_dev))
            if self.config.save_dev_pics:
                filename = os.path.join(self.config.exp_out_dir, 'dev', 'dev_tree_%s.png' % steps)
                self.ghhcTree.plot_tree(self.dev_set, filename)

            # record the best dev score to try to understand if we end up doing worse, not used at inference time
            # last model is used at inference.
            self.best_dev_dp_score = max(self.best_dev_dp_score,dp)
            self.best_dev_iter = steps if self.best_dev_dp_score == dp else self.best_dev_iter
            self.last_dev_dp_score = dp
            self.last_dev_iter = steps
            # save every time we run this eval
            self.ckpt.save(self.checkpoint_prefix)
            self.config.last_model = tf.train.latest_checkpoint(self.config.checkpoint_dir)
            if self.best_dev_dp_score == dp:
                self.config.best_model = tf.train.latest_checkpoint(self.config.checkpoint_dir)
            self.config.save_config(self.config.exp_out_dir, filename='config.json')
            return dp
        else:
            return 0.0

    def inference(self, indexes, dataset, batch_size=1000, episode_size=5000):
        batches_so_far = 0
        curr_idx = 0
        episode_size = self.config.episode_size
        if self.config.shuffle:
            indexes = indexes[np.random.permutation(indexes.shape[0]), :]
        for i in range(self.config.num_iterations):
            if curr_idx > indexes.shape[0]:
                logging.info('Restarting....')
                curr_idx = 0
                if self.config.shuffle:
                    indexes = indexes[np.random.permutation(indexes.shape[0]), :]
            logging.info('Starting iteration %s of %s' % (i, self.config.num_iterations))
            batches_so_far += self.episode_inference(indexes[curr_idx:(curr_idx+episode_size), 0],
                                                                                             indexes[curr_idx:(curr_idx+episode_size), 1],
                                                                                             indexes[curr_idx:(curr_idx+episode_size), 2],
                                                                                             dataset, batch_size, examples_so_far=batches_so_far)


