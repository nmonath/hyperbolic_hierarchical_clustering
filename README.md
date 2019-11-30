
# gHHC

Code for: [Gradient-based Hierarchical Clustering using Continuous Representations of Trees in Hyperbolic Space](https://dl.acm.org/citation.cfm?id=3330997). Nicholas Monath, Manzil Zaheer, Daniel Silva, Andrew McCallum, Amr Ahmed. KDD 2019.

## Setup

In each shell session, run:

```
source bin/setup.sh
```

to set environment variables.

Install jq (if not already installed): https://stedolan.github.io/jq/

Install maven (if not already installed):

```
sh bin/install_mvn.sh
```

Install python dependencies:

```
conda create -n env_ghhc pip python=3.6
source activate env_ghhc
# Either (linux)
wget https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.12.0-cp36-cp36m-linux_x86_64.whl
pip install tensorflow-1.12.0-cp36-cp36m-linux_x86_64.whl
# or (mac)
wget https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.12.0-py3-none-any.whl
pip install tensorflow-1.12.0-py3-none-any.whl
conda install scikit-learn
conda install tensorflow-base=1.13.1
```

See [env.yml](env.yml) for a complete list of dependencies if you run into issues
with the above.

Build scala code:

```
mvn clean package
```

Note you may need to set `JAVA_HOME` and `JAVA_HOME_8` on your system.

ALOI and Glass are downloadable from: https://github.com/iesl/xcluster

Covtype is available here: https://archive.ics.uci.edu/ml/datasets/covertype

Contact me regarding the ImageNet data.

## Clustering Experiments

### Step 1. Building triples for inference

Sample triples of datapoints that will be used for inference:

On a compute machine:

```
sh bin/sample_triples.sh config/glass/build_samples.json
```

Using slurm cluster manager:

```
sh bin/launch_samples.sh config/glass/build_samples.json <partition-name-here>
```

Note the above example is for the `glass` dataset, but the same
procedure and scripts are available for all datasets.

### Step 2. Run Inference

Update the representations of the internal nodes of the tree structure.

On a compute machine:

```
sh bin/run_inf.sh config/glass/glass.json
```

Using slurm cluster manager:

```
sh bin/launch_inf.sh config/glass/glass.json <partition-name-here>
```

This will create a directory in `exp_out/dataset_name/ghhc/timestamp`
containing the internal node parameters and configs to run the next step.
For example, this would create the following:

```
exp_out/glass/ghhc/2019-11-29-20-13-29-alg_name=ghhc-init_method=randompts-tree_learning_rate=0.01-loss=sigmoid-lca_type=conditional-num_samples=50000-batch_size=500-struct_prior=pcn
```

### Step 3. Final clustering

Produce assignment of datapoints in the hierarchical clustering and
produce internal structure.

For datasets other than ImageNet:

On a compute machine:

```
# Generally:
sh bin/run_predict_only.sh exp_out/data/ghhc/timestap/config.json data/datasetname/data_to_run_on.tsv

# For example:
sh bin/run_predict_only.sh exp_out/glass/ghhc/2019-11-29-20-13-29-alg_name=ghhc-init_method=randompts-tree_learning_rate=0.01-loss=sigmoid-lca_type=conditional-num_samples=50000-batch_size=500-struct_prior=pcn/config.json data/glass/glass.tsv
```

Using slurm cluster manager:

```
sh bin/launch_predict_only.sh exp_out/glass/ghhc/2019-11-29-20-13-29-alg_name=ghhc-init_method=randompts-tree_learning_rate=0.01-loss=sigmoid-lca_type=conditional-num_samples=50000-batch_size=500-struct_prior=pcn/config.json data/glass/glass.tsv <partition-name>
```

This will create a file: `exp_out/glass/ghhc/2019-11-29-20-13-29-alg_name=ghhc-init_method=randompts-tree_learning_rate=0.01-loss=sigmoid-lca_type=conditional-num_samples=50000-batch_size=500-struct_prior=pcn/results/tree.tsv`
which can be evaluated using

```
sh bin/score_tree.sh exp_out/glass/ghhc/2019-11-29-20-13-29-alg_name=ghhc-init_method=randompts-tree_learning_rate=0.01-loss=sigmoid-lca_type=conditional-num_samples=50000-batch_size=500-struct_prior=pcn/results/tree.tsv
```


For ImageNet:

```
 sh bin/launch_predict_only_imagenet.sh exp_out/ilsvrc/ghhc/2019-11-29-08-04-23-alg_name=ghhc-init_method=randhac-tree_learning_rate=0.01-loss=sigmoid-lca_type=conditional-num_samples=50000-batch_size=100-struct_prior=pcn/config.json data/ilsvrc/ilsvrc12.tsv.1 cpu 32000
```

This assumes that the ImageNet data file has been split into 13 files:

```
data/ilsvrc/ilsvrc12.tsv.1.split_aa
data/ilsvrc/ilsvrc12.tsv.1.split_ab
...
data/ilsvrc/ilsvrc12.tsv.1.split_am
```

Then when all jobs finish, concatenate results:

```
sh bin/cat_imagenet_tree.sh exp_out/ilsvrc/ghhc/2019-11-29-08-04-23-alg_name=ghhc-init_method=randhac-tree_learning_rate=0.01-loss=sigmoid-lca_type=conditional-num_samples=50000-batch_size=100-struct_prior=pcn/results/
```

This will create a file containing the entire tree:

```
exp_out/ilsvrc/ghhc/2019-11-29-08-04-23-alg_name=ghhc-init_method=randhac-tree_learning_rate=0.01-loss=sigmoid-lca_type=conditional-num_samples=50000-batch_size=100-struct_prior=pcn/results/tree.tsv
```

which can be evaluated using:

```
sh bin/score_tree.sh exp_out/ilsvrc/ghhc/2019-11-29-08-04-23-alg_name=ghhc-init_method=randhac-tree_learning_rate=0.01-loss=sigmoid-lca_type=conditional-num_samples=50000-batch_size=100-struct_prior=pcn/results/tree.tsv
```

## Citation

```
@inproceedings{Monath:2019:GHC:3292500.3330997,
     author = {Monath, Nicholas and Zaheer, Manzil and Silva, Daniel and McCallum, Andrew and Ahmed, Amr},
     title = {Gradient-based Hierarchical Clustering Using Continuous Representations of Trees in Hyperbolic Space},
     booktitle = {Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
     series = {KDD '19},
     year = {2019},
     isbn = {978-1-4503-6201-6},
     location = {Anchorage, AK, USA},
     pages = {714--722},
     numpages = {9},
     url = {http://doi.acm.org/10.1145/3292500.3330997},
     doi = {10.1145/3292500.3330997},
     acmid = {3330997},
     publisher = {ACM},
     address = {New York, NY, USA},
     keywords = {clustering, gradient-based clustering, hierarchical clustering},
}
```

## License

Apache License, Version 2.0

## Questions / Comments / Bugs / Issues

Please contact Nicholas Monath (nmonath@cs.umass.edu).

Also, please contact me for access to the data.

