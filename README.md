# SuRe-Transformer: Sufficient and Representative Transformer
![Graphical abstract](graphical-abstract.png) 

- - -
### Dataset
* Training and internal validation: The WSIs used in training and internal validation are from TCGA (https://portal.gdc.cancer.gov/), open access to all. 
* External validation: we have collected a total of 192 breast cancer patients' data (WSI + HRD status) from the Beijing ChosenMed Clinical Laboratory Co. Ltd. Based on this dataset, we conduct external validation (train on TCGA, test on ChosenMed).
- - -

### get_DINO_features: Use DINO for patch embedding

* `get_feature.py`：Use the trained DINO for patch embedding.

### patch_clustering: Clustering patches in WSI 

* `featureClustering_dino_8K.py`：Clustering with the features extracted from DINO.
* `genFeature_multi_8K.py`：According to the WSI patch clustering results, randomly select patches to build SuRe-Transformer training set.

### Sufficient and Representative Transformer: SuRe-Transformer training

* `splits_k_fold.py`：Make and split training set, validation set and test set for SuRe-Transformer model.

* `train_k_fold_random.py`：Training the SuRe-Transformer model.

*  `test_k_fold_random.py`：Testing the SuRe-Transformer model.

- - - 
### Environments
* Python==3.8.3
* Ubuntu==18.04.5
* torch==1.7.0
* torchvision==0.8.1
* timm==0.3.2
* imageio==2.25.1
* matplotlib==3.6.2
* numpy==1.23.0
* opencv-python==4.6.0.66
* pandas==1.5.2
* Pillow==9.4.0
* scikit_image==0.17.2
* scikit_learn==1.2.1
* scipy==1.10.1
* skimage==0.0
* spams==2.6.5.4
* tensorboardX==2.6
* pytorch_lightning==1.9.3
* openslide_python==1.1.2
- - -
If you have any questions, please contact Haijing Luan at luanhaijing@cnic.cn.
