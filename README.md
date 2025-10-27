# TOHGS
Task-Oriented Human Grasp Synthesis via Context- and Task-Aware Diffusers
## Env info
- Ubuntu 20.04
- Pytorch 2.0.1
- CUDA 11.7
- Python 3.8
```
# create conda env
conda create -n tohgs python=3.8
conda activate tohgs
```

## Install
1. MANO
```
cd manopth
pip install .
```
2. Install Pytorch, Pytorch3D and Kaolin
```
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
pip install fvcore
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu117_pyt201/download.html
pip install kaolin==0.15.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.0.1_cu117.html
```
3. Other Python packages
```
pip install -r requirements.txt
```
4. PointNet++
```
git clone https://github.com/erikwijmans/Pointnet2_PyTorch.git
cd Pointnet2_PyTorch
pip install -e .
pip install pointnet2_ops_lib/.
```
5. V-hacd
```
git clone https://github.com/kmammou/v-hacd.git
cd v-hacd
cd app
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```


## Download require files
1. Regster and download MANO model from [Link](https://mano.is.tue.mpg.de/index.html)
   unzip and copy the file `MANO_RIGHT.pkl` into `TOHGS/models/mano`
2. Download object meshes file from [Link](https://huggingface.co/datasets/liuallen871219/TOHGS)
   unzip meshes.zip and copy all of folders and files into `TOHGS/models/`
3. Download dataset from [Link](https://huggingface.co/datasets/liuallen871219/TOHGS)
   unzip task_oriented_grasps_dataset.zip and copy folder `task_oriented_grasps_dataset` into `TOHGS/`
   
## Acknowledgements

This implementation is mainly based on:
- [GraspTTA](https://github.com/hwjiang1510/GraspTTA)
- [SceneDiffuser](https://github.com/scenediffuser/Scene-Diffuser)
- [manopth](https://github.com/hassony2/manopth)

Thanks to these great open-source implementations!

## Bibtex
If you find this work helpful, please consider citing our paper:
```bash
@inproceedings{liu2025task,
  title={Task-Oriented Human Grasp Synthesis via Context-and Task-Aware Diffusers},
  author={Liu, An-Lun and Chao, Yu-Wei and Chen, Yi-Ting},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={10375--10385},
  year={2025}
}
```
