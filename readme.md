# D-IF: Uncertainty-aware Human Digitization via Implicit Distribution Field 【完善的readme待补充】

## 数据集处理

请参考ICON的dataset处理方法

## 训练&测试&推理
train:
```bash
CUDA_VISIBLE_DEVICES=7 python -m apps.train -cfg ./configs/train/d_if.yaml
```
test:
```bash
python -m apps.train -cfg ./configs/train/d_if.yaml -test
```
infer:
```bash
python -m apps.infer -cfg ./configs/d_if.yaml -gpu 0 -in_dir ./examples -out_dir ./results -export_video -loop_smpl 100 -loop_cloth 200 -hps_type pixie
```