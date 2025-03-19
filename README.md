# :sparkles:[CVPR 2025] GigaHands: A Massive Annotated Dataset of Bimanual Hand Activities

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
        ├── keypoints_3d/						# 3D hand keypoints
        ├── mano_vid/							# visualizations of mano parameters 
        ├── params/							# mano parameters
        ├── rgb_vid/							# raw multview videos
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

We store our dataset on Globus. You can download a demo sequence from [here](https://g-2488dc.56197.5898.data.globus.org/demo_data.tar%2Cgz), all annotations from [here](https://g-2488dc.56197.5898.data.globus.org/demo_all.tar.gz), and access the raw data via [here](https://app.globus.org/file-manager?origin_id=1f1426dd-3440-4cae-8c57-4a0e6934eaf2&origin_path=%2F).

### Whole Dataset

To be released.

## Checklist

- [x] Release demo data
- [ ] Release inference code for text-to-motion task
- [ ] Release training code for text-to-motion task
- [ ] Release whole dataset

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

