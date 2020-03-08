# building_segmentation_cv
### Image Segmentation using Pytorch - Developed for DrivenData's Open Cities AI Distaster Relief Competition
Authors: Thomas Kavanagh and Alex Weston

This repo holds code used in our submission Driven Data's Open Cities AI challenge. We use a Resnet backbone combined with an Joint Pyramidal Upsampling module and EncNet Head. 

To reproduce our results, first run scan scenes.py to extract 1024x1024 tiles from each raster provided by Driven Data and upload them to a Google Cloud storage bucket. This may take several days, depending on processing power and internet speeds. 

Then download the training data (either locally or on a remote instance) run train.py to train the model on the downloaded tiles. After the model is done training, run evaluate.py to predict on the test set. 

## Citations

Thanks to the following authors for their research work on semantic segmentation:
```
@inproceedings{wu2019fastfcn,
  title     = {FastFCN: Rethinking Dilated Convolution in the Backbone for Semantic Segmentation},
  author    = {Wu, Huikai and Zhang, Junge and Huang, Kaiqi and Liang, Kongming and Yu Yizhou},
  booktitle = {arXiv preprint arXiv:1903.11816},
  year = {2019}
}
```
```
@InProceedings{Zhang_2018_CVPR,
author = {Zhang, Hang and Dana, Kristin and Shi, Jianping and Zhang, Zhongyue and Wang, Xiaogang and Tyagi, Ambrish and Agrawal, Amit},
title = {Context Encoding for Semantic Segmentation},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2018}
}
```
```
@inproceedings{berman2018lovasz,
  title={The Lov{\'a}sz-Softmax loss: A tractable surrogate for the optimization of the intersection-over-union measure in neural networks},
  author={Berman, Maxim and Rannen Triki, Amal and Blaschko, Matthew B},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={4413--4421},
  year={2018}
}
```
