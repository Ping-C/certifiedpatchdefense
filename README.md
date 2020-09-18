Certified Defenses for Adversarial Patches - ICLR 2020
=====================
This repository implements the _first_ certified defense method against adversarial patch attack.
Our methodology extends Interval Bound Propagation ([IBP](https://arxiv.org/abs/1810.12715)) 
 to defending against patch attack. The resulting model achieves certified accuracy 
 that exceeds empirical robust accuracy of previous empirical defense methods, such as 
 [Local Gradient Smoothing](https://arxiv.org/abs/1807.01216) or [Digital Watermarking](https://ieeexplore.ieee.org/document/8575371). More details of our methodology can be found 
 in the paper below:

[**Certified Defenses for Adversarial Patches**](https://openreview.net/forum?id=HyeaSkrYPH&noteId=HyeaSkrYPH) <br>
_Ping-yeh Chiang*, Renkun Ni*, Ahmed Abdelkader, Chen Zhu, Christoph Studor, Tom Goldstein_<br>
ICLR 2020 <br>

Reproduce Best Performing Models
---------------------
You can reproduce our best performing models against patch attack by running the following scripts. You could also download pretrained models [here](https://drive.google.com/file/d/1cw3N3M3mZ4AXS8d3psKzgMiq7U5LWowL/view?usp=sharing) <br>
```bash
python train.py --config config/cifar_robtrain_p22_guide20.json --model_subset 3
python train.py --config config/cifar_robtrain_p55_rand20.json --model_subset 3
python train.py --config config/mnist_robtrain_p22_all.json --model_subset 0
python train.py --config config/mnist_robtrain_p55_all.json --model_subset 0
``` 

The IBP method also yields good performance against sparse attack. The models can be reproduced by running the following scripts<br>
```bash
python train.py --config config/cifar_robtrain_k4_sparse.json
python train.py --config config/cifar_robtrain_k10_sparse.json
python train.py --config config/mnist_robtrain_k4_sparse.json
python train.py --config config/mnist_robtrain_k10_sparse.json
``` 

To evaluate the trained models, use `eval.py` with the same arguments
```bash
python eval.py --config config/cifar_robtrain_p22_guide20.json --model_subset 3
python eval.py --config config/cifar_robtrain_p55_rand20.json --model_subset 3
python eval.py --config config/mnist_robtrain_p22_all.json --model_subset 0
python eval.py --config config/mnist_robtrain_p55_all.json --model_subset 0
python eval.py --config config/cifar_robtrain_k4_sparse.json
python eval.py --config config/cifar_robtrain_k10_sparse.json
python eval.py --config config/mnist_robtrain_k4_sparse.json
python eval.py --config config/mnist_robtrain_k10_sparse.json
``` 
If you run into cuda memory error, you can increase the number of gpus with `--gpu` argument (e.g. `--gpu 0,1,2,3`)

Results
---------------------

|Dataset | Training Method | Model Architecture | Attack Model | Certified Accuracy | Clean Accuracy|
|:-------: | :------: | :-------: | :-------: | :-------: | :-------:|
|MNIST | All Patch | MLP | 2×2 patch | 91.51% | 98.55% |
|MNIST | All Patch | MLP | 5×5 patch | 61.85% | 93.81% |
|CIFAR | Guided Patch 20 | 5-layer CNN | 2×2 patch | 53.02% | 66.50% |
|CIFAR | Random Patch 20 | 5-layer CNN | 5×5 patch | 30.30% | 47.80% |
|MNIST | Sparse | MLP | sparse k=4 | 90.70% | 97.20% |
|MNIST | Sparse | MLP | sparse k=10 | 75.60% | 94.64% |
|CIFAR | Sparse | MLP | sparse k=4  | 32.70% | 49.82% |
|CIFAR | Sparse | MLP | sparse k=10  | 28.21% | 44.34% |

References
---------------------
Sven Gowal, Krishnamurthy Dvijotham, Robert Stanforth, Rudy Bunel, Chongli Qin, Jonathan Uesato, Timothy Mann, and Pushmeet Kohli. "On the effectiveness of interval bound propagation for training verifiably robust models." arXiv preprint arXiv:1810.12715 (2018).

Huan Zhang, Hongge Chen, Chaowei Xiao, Sven Gowal, Robert Stanforth, Bo Li, Duane Boning, Cho-Jui Hsieh "Towards Stable and Efficient Training of Verifiably Robust Neural Networks" arXiv preprint arXiv:1906.06316 (2019)


Citation
---------------------
```bash
@inproceedings{
    Chiang2020Certified,
    title={Certified Defenses for Adversarial Patches},
    author={Ping-yeh Chiang* and Renkun Ni* and Ahmed Abdelkader and Chen Zhu and Christoph Studor and Tom Goldstein},
    booktitle={International Conference on Learning Representations},
    year={2020},
    url={https://openreview.net/forum?id=HyeaSkrYPH}
}
``` 

