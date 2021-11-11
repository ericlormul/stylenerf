## modified StyleNeRF

The code is based on the idea of [StyleNeRF](https://arxiv.org/abs/2110.08985)

## Code Structure
- Model configs: train_nerf.py
- Generator: nerf.py
- StyleNeRF implementation: nerf_networks.py
- Camera: camera.py
- Ray generation: nerf_sample_ray_split.py

## Entry Point

CUDA_VISIBLE_DEVICES=1 python train_nerf.py --outdir=~/training-runs --cfg=stylegan3-r --data=../data/metfaces-256x256.zip --gpus=1 --batch=32 --gamma=6.6

## Visualize camera poses

pip install open3d<br>
python visualize_cameras.py

