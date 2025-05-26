# OPONeRF: One-Point-One NeRF for Robust Neural Rendering

## TODO List
- [x] Dataset Release
  - [x] Cooking-Perturbation Dataset
  - [x] MeetRoom-Perturbation Dataset
  - [x] Spatial Reconstruction Dataset Links
  - [x] Spatio-temporal Reconstruction Dataset Links
- [x] Code Release
  - [x] Training Code
  - [x] Inference Code


## Usage
### Setup
```shell
pip install -r requirements.txt
```
### Run
```shell
bash train_beef.sh
```

### Static Scene Datasets
* **Cooking-Perturbation Dataset** ([Download](https://cloud.tsinghua.edu.cn/d/90ef1341cd0c4cfebb37/))
  - Re-organized from N3DV dataset
  - Camera poses estimated using COLMAP

* **MeetRoom-Perturbation Dataset** ([Download](https://cloud.tsinghua.edu.cn/d/fa1bc48c5bfd4c62a87f/))
  - Re-organized from MeetRoom dataset
  - Camera poses estimated using COLMAP

### Generalization Datasets 
* **Generalizable Spatial Reconstruction**
  - Available via [IBRNet](https://github.com/googleinterns/IBRNet) or [NeuRay](https://github.com/liuyuan-pal/NeuRay)

* **Generalizable Spatio-temporal Reconstruction**
  - Available via [MonoNeRF](https://github.com/tianfr/MonoNeRF)

The poses of Cooking-Perturbation and MeetRoom-Perturbation were estimated via [colmap](https://github.com/yzheng97/OPONeRF). 

OPONeRF
|-- data
    |--model
        |-- neuray_gen_depth_train_beef_hyper
    |-- example
        |-- beef_0
        |-- beef_1
        ...
        |-- beef_21
    |-- render
        |-- example
            |-- beef_0
            |-- beef_1
            ...
            |-- beef_21
```

## Citation
If you find our work useful in your research, please consider citing:
```
@article{zheng2024oponerf,
  title={OPONeRF: One-Point-One NeRF for Robust Neural Rendering},
  author={Zheng, Yu and Duan, Yueqi and Zheng, Kangfu and Yan, Hongru and Lu, Jiwen and Zhou, Jie},
  journal={arXiv preprint arXiv:2409.20043},
  year={2024}
}
