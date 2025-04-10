# E-HRSAI

The source code of our **Information Fusion 2025** paper *"High-Resolution Synthetic Aperture Imaging Method and Benchmark Based on Event-Frame Fusion"*.



## Requirements

1. Python 3.8 with the following packages installed:
   * opencv-python==4.6.0.66
   * torch==1.9.0
   * pillow==10.4.0
   * prefetch_generator==1.0.1
2. CUDA
   
   - **CUDA** enabled **GPUs** are required for training. We train and test our code with CUDA 11.1 V11.1.105 on A100 GPUs.
   
     

## Dataset

1. Our $\text{THU}^\text{E-HRSAI}$ dataset could be downloaded from https://github.com/lisiqi19971013/event-based-datasets. 

2. Download the pre-trained model from https://drive.google.com/file/d/1Gx0zhIeciHGEqrRryPmAC-mqoNO1wuMQ/view?usp=sharing or from https://pan.baidu.com/s/1ONvkUyk2cqWM2XR_XwaKeg (extract code: 2024).

   

## Evaluation

1. Run the following code to generate HR SAI results.

   ```shell
>>> python eval.py --folder "dataset folder" --ckpt "checkpoint path" --opFolder "opFolder"
   ```

   
   Then, the outputs will be saved in "opFolder".
   
2. Calculate metrics using the following code.

   ```shell
   >>> python calMetric.py --opFolder "opFolder" --dataFolder "dataset folder"
   ```

   The quantitative results will be save in "opFolder/res.txt"

3. Downstream applications. Change the folder to "./ultralytics". Download the checkpoint "**yolov8x.pt**" from the YOLOv8 official repository, *e.g.*, https://docs.ultralytics.com/models/yolov8/. Then, run the following code.

   ```shell
   >>> python test.py --ipFolder "HRSAI output folder" --opFolder "output detection folder" --ckptPath "yolo checkpoint path"
   ```

   For depth estimation, directly use the NeWCRFs official code.

   

## Citation

```bib
@article{hrsai,
    title={High-Resolution Synthetic Aperture Imaging Method and Benchmark Based on EventFrame Fusion}, 
    author={Li, Siqi and Li, Yipeng and Liu, Yu-Shen and Du, Shaoyi and Yong, Jun-Hai and Gao, Yue},
    journal={Information Fusion}, 
    volume={},
    number={},
    pages={},
    year={2025},
}
```

