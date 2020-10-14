# Scripts for training adversarially robust classification models

This set of scripts allows train and evaluate robust to PGD attack classification
models on ImageNet dataset. Implementations of models are taken from the pytorchcv library,
which means robust weights can be substituted to existing scripts that rely on pytorchcv.

## Trained models

All the models are trained on adversarial samples obtained after 3 iterations of PGD with step=2/3*eps.

Model            | Input Resolution | Params(M) | MACs(G) | eps | dist | Top-1 error | Top-5 error | Top-1 adv error | Top-5 adv error
---              |---               |---        |---      |---  |---   |---          |---          |---              |---
[MobilenetV2 1x](https://drive.google.com/file/d/1WCRjp9Q1oIuRpjmu9hLuE8uf9pLV2LMN/view?usp=sharing)   |224x224           | 3.4       | 0.3     | 0.02| l2   | 72.16       | 90.62       | 71.72           | 90.40
[MobilenetV2 1x](https://drive.google.com/file/d/1O82imwnSBfiaLRFs361jXWmgwkjWwYGv/view?usp=sharing)   |224x224           | 3.4       | 0.3     | 0.05| l2   | 72.12       | 90.34       | 71.11           | 89.84
[MobilenetV2 1x](https://drive.google.com/file/d/1Cz89u3J-0yrx8v8LGP98c4xQw3Dz5sfe/view?usp=sharing)   |224x224           | 3.4       | 0.3     | 0.3 | l2   | 71.38       | 89.8        | 68.79           | 88.50
