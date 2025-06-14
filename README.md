# :sparkles:[CVPR 2025 Highlight] GigaHands: A Massive Annotated Dataset of Bimanual Hand Activities

[CVPR 2025] Official repository of "GigaHands: A Massive Annotated Dataset of Bimanual Hand Activities".

[[Project Page]](https://ivl.cs.brown.edu/research/gigahands.html) [[Paper]](https://www.arxiv.org/abs/2412.04244) [[Video]](https://ivl.cs.brown.edu/assets/images/projects/gigahands/gigahands_explain.mp4)

<p> <strong>Authors</strong>:
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

## Data Format

### Demo Data

The demo data contains 5 motion sequences. The file directory looks like this:

```
demo_data/
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

We store our dataset on Globus. You can download a demo sequence from [here](https://g-852369.56197.5898.data.globus.org/gigahands_demo.tar.gz), all annotations from [here](https://g-852369.56197.5898.data.globus.org/gigahands_demo_all.tar.gz), and access the raw data via [here](https://app.globus.org/file-manager?origin_id=d7b33299-4380-49be-9727-78271911d231&origin_path=%2F).

### Whole Dataset

[2025/05/23] For **object poses**, access our Globus repository: [here](https://app.globus.org/file-manager?origin_id=d7b33299-4380-49be-9727-78271911d231&origin_path=%2Fobject_poses%2F). Download each `.tar.gz` separately (contains 1000 motion sequences per file.)

[2025/04/30] For **multiview RGB videos**, access our Globus repository: [here](https://app.globus.org/file-manager?origin_id=d7b33299-4380-49be-9727-78271911d231&origin_path=%2Fmultiview_rgb_vids%2F). Download each `.tar.gz` separately (contains 10 views per file, 51 camera views in total.)

[2025/04/02] We are pleased to release our full **hand pose** dataset, available for download [here](https://g-852369.56197.5898.data.globus.org/hand_poses.tar.gz) (Including all `keypoints_3d`,  `keypoints_3d_mano` and `params`). 

Complete **text annotation** are available [here](https://g-852369.56197.5898.data.globus.org/annotations_v2.jsonl?download=1). We used the `rewritten_annotation` for model training. 

More data coming soon! 🔜

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

## Installation

This code requires:

* Python 3.8
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

## Acknowledgement

We appreciate helps from :  

* Public code like [EasyMocap](https://github.com/zju3dv/EasyMocap), [text-to-motion](https://github.com/EricGuo5513/text-to-motion), [TM2T](https://github.com/EricGuo5513/TM2T), [MDM](https://github.com/GuyTevet/motion-diffusion-model), [T2M-GPT](https://github.com/Mael-zys/T2M-GPT) etc.
*  This research was supported by AFOSR grant FA9550-21 1-0214, NSF CAREER grant #2143576, and ONR DURIP grant N00014-23-1-2804. We would like to thank the Ope nAI Research Access Program for API support and extend our gratitude to Ellie Pavlick, Tianran Zhang, Carmen Yu, Angela Xing, Chandradeep Pokhariya, Sudarshan Harithas, Hongyu Li, Chaerin Min, Xindi Qu, Xiaoquan Liu, Hao Sun, Melvin He and Brandon Woodard.

## License

This dataset is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.

[![CC BY-NC 4.0](https://licensebuttons.net/l/by-nc/4.0/88x31.png)](https://creativecommons.org/licenses/by-nc/4.0/)

To view a copy of this license, visit https://creativecommons.org/licenses/by-nc/4.0/.

## Citation

If you find our work useful in your research, please consider citing:

```
@article{fu2024gigahands,
  title={GigaHands: A Massive Annotated Dataset of Bimanual Hand Activities},
  author={Fu, Rao and Zhang, Dingxi and Jiang, Alex and Fu, Wanjia and Funk, Austin and Ritchie, Daniel and Sridhar, Srinath},
  journal={arXiv preprint arXiv:2412.04244},
  year={2024}
}
```
