# Code for Establishing Continuous 2D-3D Surface Correspondences for Cloth-Changing ReID
To help you better understand the CSCL framework presented in our paper, we provide codes for establishing reliable continuous 2D-to-3D mapping for pedestrian images in both general and cloth-changing ReID datasets, which corresponds to the key CSE module in our paper. 
## Setup
* Python 3.8
* Pytorch 1.11.0
Run `python setup.py install` to compile this project.
## Datasets
General ReID datasets:
- [Market1501](https://zheng-lab.cecs.anu.edu.au/Project/project_reid.html)
- [DukeMTMC-ReID](https://arxiv.org/abs/1609.01775)

Cloth-Changing ReID datasets:
- [LTCC](https://naiq.github.io/LTCC_Perosn_ReID.html)
- [PRCC](https://arxiv.org/abs/2002.02295)
- [VC-clothes](https://arxiv.org/abs/2003.04070) 
- [DP3D](Release)

Download the dataset to your local folder. For example, './data/LTCC'
## Models
### ImageNet Classification Models
- [ResNet](https://arxiv.org/abs/1512.03385)
- [DenseNet](https://arxiv.org/abs/1608.06993)
- [Inception-ResNet-V2](https://arxiv.org/abs/1602.07261)
- [Inception-V4](https://arxiv.org/abs/1602.07261)
- [Xception](https://arxiv.org/abs/1610.02357)
### Image Segmentation Models
- [UNet](https://arxiv.org/pdf/1505.00459.pdf)
- [Graphonomy](https://arxiv.org/abs/2101.10620)

### ReID-Specific models
- [MuDeep](https://arxiv.org/abs/1709.05165)
- [ResNet-mid](https://arxiv.org/abs/1711.08106)
- [HACNN](https://arxiv.org/abs/1802.08122)
- [PCB](https://arxiv.org/abs/1711.09349)
- [MLFN](https://arxiv.org/abs/1803.09132)


### Train
CSE training codes are implemented in
- `trainer/cse_trainer.py`: train CSE module for ReID models 


To train CSE module, you can do
```bash
python engine/cse_trainer.py \
--dataset './data/DP3D' \
--epoch 40 
```

