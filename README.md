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
2. Install Pytorch and Pytorch3D
```
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
pip install -r requirements.txt
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu117_pyt201/download.html
```
3. Other Python packages
```
pip install -r requirements.txt
```
4. PointNet++
```
cd Pointnet2_PyTorch
pip install pointnet2_ops_lib/.
```
5. V-hacd
```
cd v-hacd
cd app
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```


## Download require files
1. Download MANO model from [Link]()
   unzip and copy the file `MANO_RIGHT.pkl` into `TOHGS/models/mano`
2. Download object meshes file from [Link]()
   unzip and copy all of folders into `TOHGS/models/`
3. Download dataset from [Link]()
   unzip and copy folder `task_oriented_grasps_dataset` into `TOHGS/`


