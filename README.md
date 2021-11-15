# Boundary-aware Transformers for Skin Lesion Segmentation

## Introduction

This is an official release of the paper **Boundary-aware Transformers for Skin Lesion Segmentation**.

> [**Boundary-aware Transformers for Skin Lesion Segmentation**](https://arxiv.org/abs/2110.03864),  
> **Jiacheng Wang**, Yueming Jin, Shuntian Cai, Hongzhi Xu, Pheng-Ann Heng, Jing Qin, Liansheng Wang  
> In: Medical Image Computing and Computer Assisted Intervention (MICCAI), 2021  
> [[arXiv](https://arxiv.org/abs/2110.03864)][[Bibetex](https://github.com/jcwang123/BA-Transformer#citation)]

<div align="center" border=> <img src=framework.jpg width="400" > </div>

## News

- **[11/15 2021] We have released the point map data.**
- **[11/08 2021] We have released the training / testing codes.**

## Code List

- [x] Network
- [x] Pre-processing
- [x] Training Codes
- [ ] MS

For more details or any questions, please feel easy to contact us by email ^\_^

## Results

## Usage

### Dataset

Please download the dataset from [ISIC](https://www.isic-archive.com/) challenge.

### Pre-processing

Please run:

```bash
$ python src/ISBI_resize.py
$ python src/point_gen.py
```

### Point Maps

For your convenience, we release the processed maps and the dataset division.

Please download them from [Baidu Disk](https://pan.baidu.com/s/1pNbH5zUI8Dw_ZAC8Iq9f7w). Code：kmqr.

The file names are equal to the original image names.

### Training and testing

## TODO

1. We will improve the network to give a more powerful and simple lesion segmentation framework.

2. The weights will be uploaded before next month.

## Citation

If you find BAT useful in your research, please consider citing:

```
@inproceedings{wang2021boundary,
  title={Boundary-Aware Transformers for Skin Lesion Segmentation},
  author={Wang, Jiacheng and Wei, Lan and Wang, Liansheng and Zhou, Qichao and Zhu, Lei and Qin, Jing},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={206--216},
  year={2021},
  organization={Springer}
}
```
