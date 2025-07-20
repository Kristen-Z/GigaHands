<div align="center">
<h1>[CVPR 2025 Highlight] GigaHands: A Massive Anotated Dataset of Bimanual Hand Activities</h1>

<a href="https://ivl.cs.brown.edu/research/gigahands.html"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>
<a href="https://www.arxiv.org/abs/2412.04244" target="_blank" rel="noopener noreferrer"> <img src="https://img.shields.io/badge/Paper-VGGT" alt="Paper PDF">
</a>
<a href="https://ivl.cs.brown.edu/assets/images/projects/gigahands/gigahands_explain.mp4"> <img src="https://img.shields.io/badge/Demo-blue" alt="Demo">

**[Interactive 3D Vision & Learning Lab, Brown University](https://ivl.cs.brown.edu/)**
<p>
    <a href="https://freddierao.github.io/">Rao Fu<sup>*</sup></a>
    ·
    <a href="https://kristen-z.github.io/">Dingxi Zhang<sup>*</sup></a>   
    ·
    <a href="https://www.alex-jiang.com/about/">Alex Jiang</a> 	 
    ·
    <a href="https://wanjia-fu.com/">Wanjia Fu</a>          
    ·
    <a href="https://austin-funk.github.io/">Austin Funk</a> 
    ·
    <a href="https://dritchie.github.io/">Daniel Ritchie</a> 
    ·
    <a href="https://cs.brown.edu/people/ssrinath/">Srinath Sridhar</a>
</p>

<img src="./assets/teaser.jpg" alt="[Teaser Figure]" style="zoom:80%;" />
</div>

## Updates
- [2025/05/23] For **object poses**, access our Globus repository: [here](https://app.globus.org/file-manager?origin_id=d7b33299-4380-49be-9727-78271911d231&origin_path=%2Fobject_poses%2F). Download each `.tar.gz` separately (contains 1000 motion sequences per file.)

- [2025/04/30] For **multiview RGB videos**, access our Globus repository: [here](https://app.globus.org/file-manager?origin_id=d7b33299-4380-49be-9727-78271911d231&origin_path=%2Fmultiview_rgb_vids%2F). Download each `.tar.gz` separately (contains 10 views per file, 51 camera views in total.)

- [2025/04/02] We are pleased to release our full **hand pose** dataset, available for download [here](https://g-852369.56197.5898.data.globus.org/hand_poses.tar.gz) (Including all `keypoints_3d`,  `keypoints_3d_mano` and `params`). 

Complete **text annotation** are available [here](https://g-852369.56197.5898.data.globus.org/annotations_v2.jsonl?download=1). We used the `rewritten_annotation` for model training. 

More data coming soon! 🔜

## Overview
Gigahands is an extensive, fully-annotated dataset of bimanual activitites captured with 51 sycrhonized cameras. Every frame includes precise 3D hand shape & pose for both hands, 3D object shape & pose, per-pixel segmentation amsks, multi-view RGB, and calibrated camera paramteters.

## Installation

This code requires:

* Python 3.8
* conda3 or miniconda3
* CUDA capable GPU (one is enough)

### Clone & Environment
First, clone the repository to your local machine.

```shell
git clone https://github.com/Kristen-Z/GigaHands.git
```

Then, install all the necessary dependencies.
```shell
cd GigaHands
conda create -n gigahands python==3.8
conda activate gigahands
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -c conda-forge ffmpeg
pip install -r requirements.txt
```

### Build Third-Party Dependencites

#### EasyMocap
1. Create a placeholder folder and pull the toolbox from a git repository. 
```shell
mkdir -p third-party
git clone https://github.com/zju3dv/EasyMocap.git third-party/EasyMocap
```

2. Build the C/C++ extensions to link EasyMocap into your conda environment. 
```shell
cd third-party/EasyMocap
python setup.py develop
```

#### MANO Hand Model Files
1. Sign the official [MANO license](https://mano.is.tue.mpg.de/) (free for research use).
2. Navigate to the **Download** tab and download the file under Models & Code.
3. Inside the downloaded folder, locate the `MANO_*.pkl` files (`MANO_RIGHT_v1_2.pkl` and `MANO_LEFT_v1_2.pkl`).
4. Manually create a `smplh` folder inside the `body_models` folder and place both files into the new folder. The final paths should look like:
```shell
GigaHands/body_models/smplh/MANO_RIGHT_v1_2.pkl
GigaHands/body_models/smplh/MANO_LEFT_v1_2.pkl
```

#### COLMAP (optional)
Download COLMAP for ground-truth comparisons by running `brew install colmap`

### Download Pre-trained Models
Download the pretrained models by running `bash dataset/download_pretrained_models.sh`, which should be like:

```shell
./checkpoints/GigaHands/
./checkpoints/GigaHands/GPT/			# Text-to-motion generation model
./checkpoints/GigaHands/VQVAE/ 			# Motion autoencoder
./checkpoints/GigaHands/text_mot_match/		# Motion & Text feature extractors for evaluation
```

## Data Format

### Demo Data

The demo data contains 5 motion sequences. We store our dataset on Globus. You can download a demo sequence from [here](https://g-852369.56197.5898.data.globus.org/gigahands_demo.tar.gz), all annotations from [here](https://g-852369.56197.5898.data.globus.org/gigahands_demo_all.tar.gz), and access the raw data via [here](https://app.globus.org/file-manager?origin_id=d7b33299-4380-49be-9727-78271911d231&origin_path=%2F).

The file directory looks like this:

```
gigahands_demo/
├── hand_pose/
    ├── p<participant id>-<scene>-<squence id>/
        ├── bboxes/							# bounding boxes for 2D keypoints tracking
        ├── keypoints_2d/						# 2D hand keypoints 
        ├── keypoints_3d/						# 3D hand keypoints (triangulate multi-view 2D keypoints.)
        ├── keypoints_3d_mano/						# 3D hand keypoints (extract from mano parms and normalized, more smooth)
        ├── mano_vid/							# visualizations of mano parameters 
        ├── params/							# mano parameters
        ├── rgb_vid/							# raw multiview videos
        	├── brics-odrind-<camera id>-camx
        		├── xxx.mp4
        		├── xxx.txt
        	├── ...
        ├── repro_2d_vid/						# visualizations of 2d hand keypoints
        ├── repro_3d_vid/						# visualizations of 3d hand keypoints
        ├── optim_params.txt						# camera parameters
    ├── ...
└── object_pose
    ├── p<participant id>-<scene>-<squence id>/
        ├── mesh							# reconstructed object mesh
        ├── pose							# object pose
        ├── render							# visualizations of object pose
        ├── segmentation						# segmented object frames
    ├── ...
```

### Whole Dataset

The dataset directory should look like this:

```python
./dataset/GigaHands/
├── hand_poses/
    ├── p<participant id>-<scene>/
        ├── keypoints_3d/						# 3D hand keypoints (triangulate multi-view 2D keypoints.)
        ├── keypoints_3d_mano/						# 3D hand keypoints (extract from mano parms and normalized, more smooth)
        ├── params/							# mano parameters
├── object_poses/
	├── <object name>
		├── p<participant id>-<scene>_<squence id>/
			├── pose					# object 6DoF poses
└── annotations_v2.jsonl						# text annotations
```

## Visualizations

After downloading all hand pose annotations, run the script below to visualize them. 

```bash
python visualize_hands.py
```

You will see videos of the MANO render results and reprojected keypoints in the `visualizations` directory.

## Inference

Sampling results from customized descriptions:

```bash
python gen_motion_custom.py --resume-pth ./checkpoints/GigaHands/VQVAE/net_last.pth --resume-trans ./checkpoints/GigaHands/GPT/net_best_fid.pth --input-text ./input.txt
```

## Train

The results are saved in the folder `output`.

**Training motion VQ-VAE**:

```bash
python3 train_vq_hand.py \
--batch-size 256 \
--lr 2e-4 \
--total-iter 300000 \
--lr-scheduler 200000 \
--nb-code 512 \
--down-t 2 \
--depth 3 \
--dilation-growth-rate 3 \
--out-dir output \
--dataname GigaHands \
--vq-act relu \
--quantizer ema_reset \
--loss-vel 0.5 \
--recons-loss l1_smooth \
--exp-name VQVAE \
--window-size 128
```

**Training T2M GPT model**:

```bash
python3 train_t2m_trans_hand.py  \
--exp-name GPT \
--batch-size 128 \
--num-layers 9 \
--embed-dim-gpt 1024 \
--nb-code 512 \
--n-head-gpt 16 \
--block-size 51 \
--ff-rate 4 \
--drop-out-rate 0.1 \
--resume-pth output/VQVAE/net_last.pth \
--vq-name VQVAE \
--out-dir output \
--total-iter 300000 \
--lr-scheduler 150000 \
--lr 0.0001 \
--dataname GigaHands \
--down-t 2 \
--depth 3 \
--quantizer ema_reset \
--eval-iter 10000 \
--pkeep 0.5 \
--dilation-growth-rate 3 \
--vq-act relu \
```

## Checklist

- [x] Release demo data
- [x] Release hand pose data
- [x] Release multi-view video data
- [ ] Release object pose data (13k) and meshes
- [x] Release inference code for text-to-motion task
- [x] Release training code for text-to-motion task

## Citation

If you find our work useful in your research, please cite:

```
@article{fu2024gigahands,
  title={GigaHands: A Massive Annotated Dataset of Bimanual Hand Activities},
  author={Fu, Rao and Zhang, Dingxi and Jiang, Alex and Fu, Wanjia and Funk, Austin and Ritchie, Daniel and Sridhar, Srinath},
  journal={arXiv preprint arXiv:2412.04244},
  year={2024}
}
```

## Acknowledgement

GigaHands builds on excellent open‑source projects including EasyMocap, SMPL‑X/MANO, and COLMAP. This research was supported by AFOSR grant FA9550-21 1-0214, NSF CAREER grant #2143576, and ONR DURIP grant N00014-23-1-2804. We would like to thank the OpenAI Research Access Program for API support and extend our gratitude to Ellie Pavlick, Tianran Zhang, Carmen Yu, Angela Xing, Chandradeep Pokhariya, Sudarshan Harithas, Hongyu Li, Chaerin Min, Xindi Qu, Xiaoquan Liu, Hao Sun, Melvin He and Brandon Woodard.

## License

GigaHands is released under the Creative Commons Attribution-NonCommercial 4.0 International License.  See the [LICENSE](https://creativecommons.org/licenses/by-nc/4.0/) file for details.

[![CC BY-NC 4.0](https://licensebuttons.net/l/by-nc/4.0/88x31.png)](https://creativecommons.org/licenses/by-nc/4.0/)

