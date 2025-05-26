# OPONeRF

## Usage
### Setup
```shell
pip install -r requirements.txt
```
### Run
```shell
bash train_beef.sh
```

## Dataset

* [Cooking-Perturbation](https://cloud.tsinghua.edu.cn/d/90ef1341cd0c4cfebb37/) (Re-organized from N3DV)
* [MeetRoom-Perturbation](https://cloud.tsinghua.edu.cn/d/fa1bc48c5bfd4c62a87f/) (Re-organized from MeetRoom)
* [Generalizable Spatial Reconstruction](https://github.com/googleinterns/IBRNet) (or [here](https://github.com/liuyuan-pal/NeuRay), LLFF & NeRF Synthetic & Spaces & DTU & RealEstate10K)
* [Generalizable Spatio-temporal Reconstruction](https://github.com/tianfr/MonoNeRF) (Dynamic Scenes)

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

