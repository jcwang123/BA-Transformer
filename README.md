# Boundary-aware Transformers for Skin Lesion Segmentation

## Introduction

This is an official release of the paper **Boundary-aware Transformers for Skin Lesion Segmentation**.

> [**Boundary-aware Transformers for Skin Lesion Segmentation**](https://arxiv.org/abs/2110.03864),   <br/>
> **Jiacheng Wang**, Lan Wei, Liansheng Wang, Qichao Zhou, Lei Zhu, Jing Qin <br/>
> In: Medical Image Computing and Computer Assisted Intervention (MICCAI), 2021  <br/>
> [[arXiv](https://arxiv.org/abs/2110.03864)][[Bibetex](https://github.com/jcwang123/BA-Transformer#citation)]

<div align="center" border=> <img src=framework.jpg width="400" > </div>

## News
- **[5/27 2022] We have released a more powerful [XBound-Former](https://github.com/jcwang123/xboundformer) with clearer concept and codes.**
- **[11/15 2021] We have released the point map data.**
- **[11/08 2021] We have released the training / testing codes.**

## Code List

- [x] Network
- [x] Pre-processing
- [x] Training Codes
- [ ] MS

For more details or any questions, please feel easy to contact us by email (jiachengw@stu.xmu.edu.cn).


## Usage

### Dataset

Please download the dataset from [ISIC](https://www.isic-archive.com/) challenge and [PH2](https://www.fc.up.pt/addi/ph2%20database.html) website.

### Pre-processing

Please run:

```bash
$ python src/process_resize.py
$ python src/process_point.py
```

You need to change the **File Path** to your own.

### Point Maps

For your convenience, we release the processed maps and the dataset division.

Please download them from [Baidu Disk](https://pan.baidu.com/s/1pNbH5zUI8Dw_ZAC8Iq9f7w) (code：**kmqr**) or [Google Drive](https://drive.google.com/file/d/1mSLt-ipLM9CxrfvwgjJr5V9NKrpnQaQ5/view?usp=sharing)

The file names are equal to the original image names.

### Training 

### Testing

Download the pretrained weight for PH2 dataset from [Google Drive](https://drive.google.com/file/d/1-eMHYX1fr-QvI3n50S0xqWcxc3FGsMgE/view?usp=sharing).

```bash
$ python test.py --dataset isic2016
```

### Result

|Method | Dice | IoU | HD95 |  ASSD|
| ------ | ------ | ------ |------ |------ |
Lee *et al.* | 0.918 | 0.843 | - | - |
BAT (paper)| 0.921 | 0.858 | - | - |
<!-- BAT (val.)| 0.918 | 0.854 | **6.664** | **0.923** | -->
<!-- BAT (w/o val.)| **0.923** | **0.864** | 7.515 | 1.043 | -->


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
