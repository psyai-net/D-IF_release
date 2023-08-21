![Psyche AI Inc release](./media/psy_logo.png)

# D-IF: Uncertainty-aware Human Digitization via Implicit Distribution Field [ICCV2023]

Official PyTorch implementation for the paper:

> **D-IF: Uncertainty-aware Human Digitization via Implicit Distribution Field**, ***ICCV 2023***.
>
> Xueting Yang, Yihao Luo, Yuliang Xiu, Wei Wang, Hao Xu, Zhaoxin Fan
>
> <a href='https://arxiv.org/abs/2308.08857'><img src='https://img.shields.io/badge/arXiv-2308.08857-red'></a> <a href='https://yxt7979.github.io/idf/'><img src='https://img.shields.io/badge/Project-Video-Green'></a> [![License ↗](https://img.shields.io/badge/License-CCBYNC4.0-blue.svg)](LICENSE)


<p align="center">
<img src="./media/DIF-pipeline .png" width="90%" />
</p>

> Detailed human reconstruction from a single image using Implicit Distribution Field.

## Environment

- Linux
- Python 3.8
- Pytorch 1.13.0
- CUDA 11.3
- CUDA=11.3, GPU Memory > 12GB
- PyTorch3D

Clone the repo:
  ```bash
  git clone https://github.com/psyai-net/D-IF_release.git
  cd D-IF_release
  ```  
Create conda environment:
```bash
conda env create -f environment.yaml
conda init bash
source ~/.bashrc
source activate D-IF
pip install -r requirements.txt --use-deprecated=legacy-resolver
```


## **Demo**

```bash
python -m apps.infer -cfg ./configs/d_if.yaml -gpu 0 -in_dir ./examples -out_dir ./results -export_video -loop_smpl 100 -loop_cloth 200 -hps_type pixie
```

## **Train/Test**
Train dataset: Thuman2.0, for download, please follow the steps of [ICON_train](https://github.com/YuliangXiu/ICON/blob/master/docs/dataset.md#thuman20) completely.
```bash
CUDA_VISIBLE_DEVICES=7 python -m apps.train -cfg ./configs/train/d_if.yaml
```
Test dataset: CAPE,  for download, please follow the steps of [ICON_test](https://github.com/YuliangXiu/ICON/blob/master/docs/evaluation.md#cape-testset) completely.
```bash
python -m apps.train -cfg ./configs/train/d_if.yaml -test
```

## **Citation**
If you find this work useful for your research, please cite our paper:
```
comming soon
```

## **Acknowledgement**
Here are some great resources we benefit:
- [ICON](https://github.com/YuliangXiu/ICON/) for pipeline.
- [CAPE](https://github.com/QianliM/CAPE) and [THuman](https://github.com/ytrock/THuman2.0-Dataset) for datasets.
- [PaMIR](https://github.com/ZhengZerong/PaMIR), [PIFu](https://github.com/shunsukesaito/PIFu), [ECON](https://github.com/YuliangXiu/ECON/), and [ICON](https://github.com/YuliangXiu/ICON/) for Benchmark
- [PyTorch3D](https://github.com/facebookresearch/pytorch3d) for Differential Rendering.

## **Contact**
For research purpose, please contact xueting.yang99@gmail.com

For commercial licensing, please contact fanzhaoxin@psyai.net and ps-licensing@tue.mpg.de

## **License**
This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License. Please read the [LICENSE](LICENSE) file for more information.

## **Invitation**

We invite you to join [Psyche AI Inc](https://www.psyai.com/home) to conduct cutting-edge research and business implementation together. At Psyche AI Inc, we are committed to pushing the boundaries of what's possible in the fields of artificial intelligence and computer vision, especially their applications in avatars. As a member of our team, you will have the opportunity to collaborate with talented individuals, innovate new ideas, and contribute to projects that have a real-world impact.

If you are passionate about working on the forefront of technology and making a difference, we would love to hear from you. Please visit our website at [Psyche AI Inc](https://www.psyai.com/home) to learn more about us and to apply for open positions. You can also contact us by fanzhaoxin@psyai.net.

Let's shape the future together!!

