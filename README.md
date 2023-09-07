# Focal Network for Image Restoration

Yuning Cui, [Wenqi Ren](https://scholar.google.com.hk/citations?user=VwfgfR8AAAAJ&hl=zh-CN&oi=ao), [Xiaochun Cao](https://scholar.google.com.hk/citations?user=PDgp6OkAAAAJ&hl=zh-CN&oi=ao), [Alois Knoll](https://scholar.google.com.hk/citations?user=-CA8QgwAAAAJ&hl=zh-CN&oi=ao)

<!-- [![](https://img.shields.io/badge/ICLR-Paper-blue.svg)](https://openreview.net/forum?id=tyZ1ChGZIKO) -->




<!-- ## Architecture -->
<!--![](figs/pipeline.png)-->

## Installation
The project is built with PyTorch 3.8, PyTorch 1.8.1. CUDA 10.2, cuDNN 7.6.5
For installing, follow these instructions:
~~~
conda install pytorch=1.8.1 torchvision=0.9.1 -c pytorch
pip install tensorboard einops scikit-image pytorch_msssim opencv-python
~~~
Install warmup scheduler:
~~~
cd pytorch-gradual-warmup-lr/
python setup.py install
cd ..
~~~
## Evaluation

## Results
The resulting images can be downloaded [here](https://drive.google.com/drive/folders/1GWgeqDuqJmR_3wy985l6Jl_ExtC3uFI_?usp=sharing).
|Task|Dataset|PSNR|SSIM|
|----|------|-----|----|
|**Image Dehazing**|ITS|40.82|0.96|
||OTS|37.71|0.995|
||Dense-Haze|17.07|0.63|
||NH-HAZE|20.43|0.79|
||O-HAZE|25.50|0.94|
||NHR|25.35|0.969|
|**Image Desnowing**|CSD|37.18|0.99|
||SRRS|31.34|0.98|
||Snow100K|33.53|0.95|
|**Image Motion Deblurring**|GoPro|33.10|0.962|


## Citation
If you find this project useful for your research, please consider citing:
~~~
@inproceedings{cui2023focalnet,
  title={Focal Network for Image Restoration},
  author={Cui, Yuning and Ren, Wenqi and Cao, Xiaochun and Knoll, Alois},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2023}
}
~~~
## Contact
Should you have any question, please contact Yuning Cui.
