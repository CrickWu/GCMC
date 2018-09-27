#!/bin/bash

# Course
python src/run.py --dataset prc --layers 50 30 5 --iters 6000 --pos_up_ratio 10. --fold 5 --save_log --save_model --binary_graph
python src/run.py --dataset cmu --layers 80 70 5 --iters 9000 --pos_up_ratio 10. --fold 5 --save_log --save_model --binary_graph
python src/run.py --dataset caltech --layers 400 300 5 --iters 7000 --pos_up_ratio 10. --fold 5 --save_log --save_model --binary_graph
python src/run.py --dataset mit --layers 400 300 5 --iters 7000 --pos_up_ratio 10. --fold 5 --save_log --save_model --binary_graph

# Drug
python src/run.py --dataset nr --layers 26 30 5 --iters 7000 --pos_up_ratio 10. --fold 5 --save_log --save_model --binary_graph
python src/run.py --dataset gpcr --layers 90 80 5 --iters 5000 --pos_up_ratio 10. --fold 5 --save_log --save_model --binary_graph
python src/run.py --dataset ic --layers 200 100 5 --iters 5000 --pos_up_ratio 10. --fold 5 --save_log --save_model --binary_graph
python src/run.py --dataset e --layers 400 300 5 --iters 2000 --pos_up_ratio 10. --fold 5 --save_log --save_model --binary_graph

# Citation
python src/run.py --dataset cora --layers 400 300 5 --iters 6000 --pos_up_ratio 10. --fold 5 --save_log --save_model --binary_graph
python src/run.py --dataset citeseer --layers 4000 300 5 --iters 6000 --pos_up_ratio 10. --fold 5 --save_log --save_model --binary_graph

python test.release.py --dataset ml-100k --layers 800 300 5 --iters 4000 --fold 5 --save_log --save_model

