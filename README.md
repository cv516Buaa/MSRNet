# MSRNet

This repo is the implementation of ["Look in Different Views: Multi-Scheme Regression Guided Cell Instance Segmentation"](https://arxiv.org/abs/2208.08078). we refer to  [MMDetection](https://github.com/open-mmlab/mmdetection) to implement cell instance segmentation task. Many thanks to SenseTime and their excellent repos.

<table>
    <tr>
    <td><img src="PaperFigs\Fig2.png" width = "100%" alt="DS2Net"/></td>
    </tr>
</table>

## Dataset
**2018 Data Science Bowl (DSB2018)** contains a total of 670 images, and the difficulty of this dataset mainly lies in the variety of image sizes, magnifications, imaging types and cell types. You can access to this dataset from [kaggle](https://www.kaggle.com/c/data-science-bowl-2018/data).

**CA2.5** consists of 524 fluorescence images of 512×512 size, which contains some severely densely packed cell images with large differences in the brightness. You can access to this dataset from [CA2.5-Net Nuclei Segmentation Framework with a Microscopy Cell Benchmark Collection](https://link.springer.com/chapter/10.1007/978-3-030-87237-3_43).

**The Sartorius Cell Instance Segmentation (SCIS)** is from a Kaggle‘s cell instance segmentation competition recently, which focus on neuronal cell instance segmentation. This dataset consists of a total of 606 images of 520×704 size. You can access to this dataset from [kaggle](https://www.kaggle.com/competitions/sartorius-cell-instance-segmentation/data).

## MSRNet
### Install

1. requirements:
    
    python >= 3.7
        
    pytorch >= 1.5
        
    cuda >= 10.0
    
2. prerequisites: Please refer to  [MMDetection PREREQUISITES](https://mmdetection.readthedocs.io/en/latest/get_started.html); Please don't forget to install mmsegmentation with

     ```
     cd MSRNet
     
     pip install -e .
     
     chmod 777 ./tools/train.py
     
     chmod 777 ./tools/test.py
     ```

### Training

#### Task: Cell Instance Segmentation

<table>
    <tr>
    <td><img src="PaperFigs\result1.png" width = "100%" alt="cell instance segmentation result"/></td>
    </tr>
</table>
  
     cd MSRNet
     
     python tools/train.py configs/msrnet/msrnet_1x_coco.py


### Testing

#### Task: Cell Instance Segmentation
  
     cd MSRNet
     
     python tools/test.py configs/msrnet/msrnet_1x_coco.py checkpoints/dsb2018_fin4_770_622.pth --eval bbox segm


## Description of MSRNet
- https://arxiv.org/abs/2208.08078

If you have any question, please discuss with me by sending email to lyushuchang@buaa.edu.cn.

If you find this code useful please cite:
```
@misc{https://doi.org/10.48550/arxiv.2208.08078,
  doi = {10.48550/ARXIV.2208.08078},
  
  url = {https://arxiv.org/abs/2208.08078},
  
  author = {Li, Menghao and Feng, Wenquan and Lyu, Shuchang and Chen, Lijiang and Zhao, Qi},
  
  keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {Look in Different Views: Multi-Scheme Regression Guided Cell Instance Segmentation},
  
  publisher = {arXiv},
  
  year = {2022},
  
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```

# References
Many thanks to their excellent works
* [MMDetection](https://github.com/open-mmlab/mmdetection)
