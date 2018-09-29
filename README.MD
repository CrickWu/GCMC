# Graph Convolutional Matrix Completion for Bipartite Edge Prediction

## Paper
[Graph Convolutional Matrix Completion for Bipartite Edge Prediction](http://nyc.lti.cs.cmu.edu/yiming/Publications/wu-kdir18.pdf) - Yuexin Wu, Hanxiao Liu and Yiming Yang KDIR 2018. (Best Student Paper)

## Dependencies
```Python == 2.7, lasagne == 0.1, theano == 0.8, numpy, scipy, joblib, scikit-learn, nose-parameterized```

### Use `Anaconda`
If you have `Anaconda`, you can install the dependencies using the following command:
```conda create --name gcmc --file requirement.txt```

## How to Run

### Preprocess
Uncompress data using `7z x data.7z`.

### Use GPU
If you have installed `cuda` support, add `THEANO_FLAGS='device=cuda0,floatX=float32'` before executing the commnad. In experiments, multi-core cpus are more preferable (with `--n_jobs` option).

### Simple Example
```
python src/run.py --dataset cmu --layers 80 70 5 --iters 1000 --pos_up_ratio 10. --fold 1 --save_log --save_model --binary_graph
```
This will create logs and model files in `save` folder. The final result is saved in `gcmc.log`.

See `python src/run.py --help` for more option explanations.

### Full Experiment in Paper
***NOTICE: This may take LONG time. Consider running single experiments first to estimate time.***

General `--iters 1000` should be enough for all datasets. In our experiment, the exact parameters can be found in `run_all.sh`.

Execute the following command to run on all datasets.
```
bash run_all.sh
```

## Citation
If you find the repository useful for your publication, please consider citing our paper:
```
@conference{wu2018graph,
author={Yuexin Wu and Hanxiao Liu and Yiming Yang},
title={Graph Convolutional Matrix Completion for Bipartite Edge Prediction},
booktitle={Proceedings of the 10th International Joint Conference on Knowledge Discovery, Knowledge Engineering and Knowledge Management - Volume 1: KDIR,},
year={2018},
pages={51-60},
publisher={SciTePress},
organization={INSTICC},
doi={10.5220/0006900000510060},
isbn={978-989-758-330-8},
}
```
For any questions and comments, please send your email to [yuexinw@andrew.cmu.edu](mailto:yuexinw@andrew.cmu.edu).
