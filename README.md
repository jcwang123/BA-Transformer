# Boundary-aware Transformers for Skin Lesion Segmentation

Recently, transformers have been proposed as a promising tool for global context modeling by employing a powerful global attention mechanism, but one of
their main shortcomings when applied to segmentation tasks is that they cannot effectively extract sufficient local details to tackle ambiguous boundaries. We propose a novel boundary-aware transformer (BAT) to comprehensively address the challenges of automatic skin lesion segmentation.

## Code List

- [x] Network
- [ ] Pre-processing
- [ ] Training Codes

## Usage

1. you can download the dataset from [ISIC](https://www.isic-archive.com/) challenge.
2. for pre-processing the dataset, you can run:

```bash
$ python src/resize.py
```

and

```bash
$ python src/point_gen.py
```

3. before running the network, you should first download the code of [CELL_DETR](https://github.com/ChristophReich1996/Cell-DETR) into [lib](https://github.com/jcwang123/BA-Transformer/lib) and install it.

## Citation
