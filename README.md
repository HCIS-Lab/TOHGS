# TOHGS
Task-Oriented Human Grasp Synthesis via Context- and Task-Aware Diffusers
## Env info
- Ubuntu 20.04
- Pytorch 2.0.1
- CUDA 11.8
- Python 3.8
```
# create conda env
conda create -n tohgs python=3.8
conda activate tohgs
```

## Install dependence
1. MANO
```
cd manopth
pip install .
```
2. PointNet++
```
cd Pointnet2_PyTorch
pip install pointnet2_ops_lib/.
```
3. V-hacd
```
cd v-hacd
cd app
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```
4. Install Pytorch3D
```
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu118_pyt201/download.html
```

