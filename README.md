# Scripts for training adversarially robust classification models

This set of scripts allows train and evaluate robust to [PGD](https://arxiv.org/abs/1706.06083) attack classification
models on ImageNet dataset. Implementations of models are taken from the [pytorchcv](https://github.com/osmr/imgclsmob/tree/master/pytorch/pytorchcv) library,
which means robust weights can be substituted to existing scripts that rely on pytorchcv.

## Trained models

All the models are trained on adversarial samples obtained after 3 iterations of PGD with step=2/3*eps.

Model            | Input Resolution | Params(M) | MACs(G) | eps | dist | Top-1 accuracy | Top-5 accuracy | Top-1 adv accuracy | Top-5 adv accuracy
---              |---               |---        |---      |---  |---   |---          |---          |---              |---
[MobilenetV2 1x](https://drive.google.com/file/d/1WCRjp9Q1oIuRpjmu9hLuE8uf9pLV2LMN/view?usp=sharing)   |224x224           | 3.4       | 0.3     | 0.02| l2   | 72.16       | 90.62       | 71.72           | 90.40
[MobilenetV2 1x](https://drive.google.com/file/d/1O82imwnSBfiaLRFs361jXWmgwkjWwYGv/view?usp=sharing)   |224x224           | 3.4       | 0.3     | 0.05| l2   | 72.12       | 90.34       | 71.11           | 89.84
[MobilenetV2 1x](https://drive.google.com/file/d/1Cz89u3J-0yrx8v8LGP98c4xQw3Dz5sfe/view?usp=sharing)   |224x224           | 3.4       | 0.3     | 0.3 | l2   | 71.38       | 89.8        | 68.79           | 88.50


An example of training command:
```bash
python train.py \
 -a mobilenetv2_w1 \
 -b 256 \
 -d $IMAGENET_FOLDER \
 --epochs 150 \
 --lr-decay cos \
 --lr 0.05 \
 --wd 4e-5 \
 -c ./snapshots \
 --input-size 224 \
 --adv-eps 0.3 \
 --euclidean \
 -j 40
```

An example of an evaluation command:
```bash
python3 train.py \
 -a mobilenetv2_w1 \
 -d $IMAGENET_FOLDER  \
 -b 128 \
 --weight ./snapshots/model_best.pth.tar \
 --evaluate \
 --input-size 224 \
 -j 8 \
 --adv-eps 0.3 \
 --euclidean
```
