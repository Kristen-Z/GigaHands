<div align="center">
<h1> <b>[CVPR 2025 Highlight]</b> <br> GigaHands: A Massive Annotated Dataset of Bimanual Hand Activities </h1>
<a href="https://ivl.cs.brown.edu/research/gigahands.html"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>
<a href="https://www.arxiv.org/abs/2412.04244" target="_blank" rel="noopener noreferrer"> <img src="https://img.shields.io/badge/Paper-VGGT" alt="Paper PDF"></a>
<a href="https://ivl.cs.brown.edu/assets/images/projects/gigahands/gigahands_explain.mp4"> <img src="https://img.shields.io/badge/Demo-blue" alt="Demo"></a>

**[Brown IVL](https://ivl.cs.brown.edu/)**
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

## 📢 Updates

- [2025/07/21] We provide [hand-object mesh visualizer](##Visualizations
). The visualizer provides hand-object temporal alignment, and camera parameter usage. 

- [2025/07/09] For **object meshes**, you can download them [here](https://g-852369.56197.5898.data.globus.org/scans_publish.zip). 

  We also provide smoother 3D hand poses that are aligned with the object coordinate system, derived from MANO parameters — available [here](https://g-852369.56197.5898.data.globus.org/keypoints_3d_mano_align.tar.gz).
  Note: the previously provided `keypoints_3d_mano` were also generated from MANO parameters, but have been normalized and recentered to better support motion generation training.

- [2025/05/23] For **object poses**, access our Globus repository: [here](https://app.globus.org/file-manager?origin_id=d7b33299-4380-49be-9727-78271911d231&origin_path=%2Fobject_poses%2F). Download each `.tar.gz` separately (contains 1000 motion sequences per file.)

- [2025/04/30] For **multiview RGB videos**, access our Globus repository: [here](https://app.globus.org/file-manager?origin_id=d7b33299-4380-49be-9727-78271911d231&origin_path=%2Fmultiview_rgb_vids%2F). Download each `.tar.gz` separately (contains 10 views per file, 51 camera views in total.)

- [2025/04/02] We are pleased to release our full **hand pose** dataset, available for download [here](https://g-852369.56197.5898.data.globus.org/hand_poses.tar.gz) (Including all `keypoints_3d`,  `keypoints_3d_mano` and `params`). 

Complete **text annotation** are available [here](https://g-852369.56197.5898.data.globus.org/annotations_v2.jsonl?download=1). We used the `rewritten_annotation` for model training. 

More data coming soon! 🔜

## 🗒️ Overview
Understanding bimanual human hand activities is a critical problem in AI and robotics. We cannot build large models of bimanual activities because existing datasets lack the scale, coverage of diverse hand activities, and detailed annotations. We introduce GigaHands, a massive annotated dataset capturing 34 hours of bimanual hand activities from **56 subjects** and **417 objects**, totaling **14k motion clips** derived from **183 million frames** paired with **84k text annotations**. Our markerless capture setup and data acquisition protocol enable fully automatic 3D hand and object estimation while minimizing the effort required for text annotation. The scale and diversity of GigaHands enable broad applications, including text-driven action synthesis, hand motion captioning, and dynamic radiance field reconstruction.

## 📂 Data Format

We store our dataset on [Globus](https://www.globus.org/). Access the raw data via [here](https://app.globus.org/file-manager?origin_id=d7b33299-4380-49be-9727-78271911d231&origin_path=%2F).
### Demo Data

You can download 1 demo sequence from [here](https://g-852369.56197.5898.data.globus.org/gigahands_demo.tar.gz).
Or download 5 demo sequences from [here](https://g-852369.56197.5898.data.globus.org/gigahands_demo_all.tar.gz). 

<summary>Directory Structure (Click to expand)</summary>

<details>
    ```text
    gigahands_demo/
    ├── hand_pose/
    │   ├── p<participant id>-<scene>-<squence id>/
    │   │   ├── bboxes/
    │   │   ├── keypoints_2d/
    │   │   ├── keypoints_3d/
    │   │   ├── keypoints_3d_mano/
    │   │   ├── mano_vid/
    │   │   ├── params/
    │   │   ├── rgb_vid/
    │   │   │   ├── brics-odroid-<camera id>-camx/
    │   │   │       ├── xxx.mp4
    │   │   │       ├── xxx.txt
    │   │   ├── repro_2d_vid/
    │   │   ├── repro_3d_vid/
    │   │   ├── optim_params.txt
    │   ├── ...
    └── object_pose/
        ├── p<participant id>-<scene>-<squence id>/
        │   ├── mesh/
        │   ├── pose/
        │   ├── render/
        │   ├── segmentation/
        ├── ...
</details>

### Whole Dataset

The dataset directory should look like this:

```python
./dataset/GigaHands/
├── multiview_rgb_vids/
    ├── p<participant id>-<scene>/
        ├── brics-odroid-<camera id>/
            ├── brics-odroid-<camera id>_<sequence 0 timestamp>.mp4
            ├── brics-odroid-<camera id>_<sequence 1 timestamp>.mp4
            ├── ...
├── hand_poses/
    ├── p<participant id>-<scene>/
        ├── keypoints_3d/						# 3D hand keypoints (triangulate multi-view 2D keypoints.)
        ├── keypoints_3d_mano/						# 3D hand keypoints (extract from mano parms and normalized, more smooth)
        ├── params/							# mano parameters
        ├── optim_params.txt				# camera parameters
├── object_poses/
    ├── <scene name>
        ├── <object name>
            ├── p<participant id>-<scene>_<squence id>/
                ├── pose					# object 6DoF poses
├── object_meta/                            # all scanned and generated meshes
    ├── <scene name>
        ├── <object name>

└── annotations_v2.jsonl 					# text annotations
```
Downlod the multiview_rgb videos from [here](https://app.globus.org/file-manager?origin_id=d7b33299-4380-49be-9727-78271911d231&origin_path=%2Fmultiview_rgb_vids%2F), hand annotations and camera parameters from [here](https://g-852369.56197.5898.data.globus.org/hand_poses.tar.gz), smoothed 3d hand keypoints from [here](https://g-852369.56197.5898.data.globus.org/keypoints_3d_mano_align.tar.gz), object poses from [here](https://app.globus.org/file-manager?origin_id=d7b33299-4380-49be-9727-78271911d231&origin_path=%2Fobject_poses%2F), object mesh from [here](https://g-852369.56197.5898.data.globus.org/scans_publish.zip), text annotations from [here](https://g-852369.56197.5898.data.globus.org/annotations_v2.jsonl?download=1), and the original instruction script grouped by scenario, scene, activity from [here](). 

## Installation

This code requires:

* Python 3.8+
* conda3 or miniconda3
* CUDA capable GPU (one is enough)

1. Create a virtual environment and install necessary dependencies

```shell
conda create -n gigahands python==3.8
conda activate gigahands
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -c conda-forge ffmpeg
pip install -r requirements.txt
```

3. Install EasyMocap

```shell
cd third-party/EasyMocap
python setup.py develop
```

4. Download [mano](https://mano.is.tue.mpg.de/download.php) models and place the `MANO_*.pkl` files under `body_models/smplh`.
5. Download the pretrained models by running `bash dataset/download_pretrained_models.sh`, which should be like:

```shell
./checkpoints/GigaHands/
./checkpoints/GigaHands/GPT/			# Text-to-motion generation model
./checkpoints/GigaHands/VQVAE/ 			# Motion autoencoder
./checkpoints/GigaHands/text_mot_match/		# Motion & Text feature extractors for evaluation
```

## Visualizations

After downloading hand pose, object pose and object meshes, run the script below to visualize hand-object mesh. 

```bash
python render_mesh_video.py \
    --dataset_root <data_root_path> \
    --scene_name 17_instruments \
    --session_name p003-instrument \
    --seq_id 33 \
    --object_name ukelele_scan \
    --mesh_name ukelele-simplified1_1.obj \
    --render_camera brics-odroid-011_cam0 \
    --save_root visualizations

```

You will see videos of the rendered hand-object mehs in `visualizations` directory.
<video src="visualizations/17_instruments/p003-instrument_0033/output.mp4" controls width="480"></video>


Visualizer below is customized for training text-hand models. 

```bash
python visualize_hands.py
```

You will see videos of the MANO render results and reprojected keypoints in the `visualizations` directory.

## 🚀 Inference - text2motion

Sampling results from customized descriptions:

```bash
python gen_motion_custom.py --resume-pth ./checkpoints/GigaHands/VQVAE/net_last.pth --resume-trans ./checkpoints/GigaHands/GPT/net_best_fid.pth --input-text ./input.txt
```

## 🏋️ Training - text2motion

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
- [ ] Release hand-object-motion and meshes correspondence file
- [x] Release inference code for text-to-motion task
- [x] Release training code for text-to-motion task

## Citation

If you find our work useful in your research, please cite:

```
@inproceedings{fu2025gigahands,
  title={Gigahands: A massive annotated dataset of bimanual hand activities},
  author={Fu, Rao and Zhang, Dingxi and Jiang, Alex and Fu, Wanjia and Funk, Austin and Ritchie, Daniel and Sridhar, Srinath},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={17461--17474},
  year={2025}
}
```

## Acknowledgement

We appreciate helps from :  

* Public code like [EasyMocap](https://github.com/zju3dv/EasyMocap), [text-to-motion](https://github.com/EricGuo5513/text-to-motion), [TM2T](https://github.com/EricGuo5513/TM2T), [MDM](https://github.com/GuyTevet/motion-diffusion-model), [T2M-GPT](https://github.com/Mael-zys/T2M-GPT) etc.
*  This research was supported by AFOSR grant FA9550-21 1-0214, NSF CAREER grant #2143576, and ONR DURIP grant N00014-23-1-2804. We would like to thank the Ope nAI Research Access Program for API support and extend our gratitude to Ellie Pavlick, Tianran Zhang, Carmen Yu, Angela Xing, Chandradeep Pokhariya, Sudarshan Harithas, Hongyu Li, Chaerin Min, Xindi Qu, Xiaoquan Liu, Hao Sun, Melvin He and Brandon Woodard.

## License

GigaHands is released under the Creative Commons Attribution-NonCommercial 4.0 International License.  See the [LICENSE](https://creativecommons.org/licenses/by-nc/4.0/) file for details.

[![CC BY-NC 4.0](https://licensebuttons.net/l/by-nc/4.0/88x31.png)](https://creativecommons.org/licenses/by-nc/4.0/)

