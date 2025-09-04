

# Object Data Annotation (CVPR 2025 version)

We release object motion tracking results with manual validation for GigaHands dataset.


## 📂 Object Pose Data

Download object pose [here](https://app.globus.org/file-manager?origin_id=2dc33637-199d-4199-bf5d-40004da7e485&origin_path=%2Fdata%2Fssrinath%2Fdatasets%2FAction%2Fbrics-mini%2Fpublic_data%2F). There are 3.3k object motion sequences in 4 zip files ([zip 1](https://m-8f2a9c.56197.5898.data.globus.org/data/ssrinath/datasets/Action/brics-mini/public_data/pose_jsons_round1_001.zip), [zip 2](https://m-8f2a9c.56197.5898.data.globus.org/data/ssrinath/datasets/Action/brics-mini/public_data/pose_jsons_round1_002.zip), [zip 3](https://m-8f2a9c.56197.5898.data.globus.org/data/ssrinath/datasets/Action/brics-mini/public_data/pose_jsons_round1_003.zip), [zip 4](https://m-8f2a9c.56197.5898.data.globus.org/data/ssrinath/datasets/Action/brics-mini/public_data/pose_jsons_round1_004.zip)).
Tracked object poses are stored under:

```
brics-mini-objects-round1/{scene_name}/{object_name}/{sequence_id}/pose/optimized_pose.json
```

Each `optimized_pose.json` contains the estimated 3D pose of an object instance in the given scene and sequence.

> **Note on deformable and articulated objects**:
> Some objects in the dataset are **non-rigid or articulated**.
> We still provide their **6-DoF tracking** for completeness, even though it is unconventional to model them this way.
>
> For these cases, we mark tracking as **success** if the segmentation mask **IoU > 60%**, regardless of the underlying deformation or part movement.
>
> We hope this annotation will inspire further work on articulation and deformation object modeling and tracking, which remains an open challenge.

## 🧩 Mesh Data

Download object mesh files [here](https://g-852369.56197.5898.data.globus.org/scans_publish.zip). Object mesh files are stored separately under:

```
scans_publish/{scene_name}/{object_name}/mesh_obj_file
```

These meshes correspond to the `<object_path>` entries used in the annotation CSVs.


## ✅ Annotation Format

Down load object meta data [here](https://m-8f2a9c.56197.5898.data.globus.org/data/ssrinath/datasets/Action/brics-mini/public_data/object_meta.zip). Per-scene annotations are provided as CSV files, with both **object presence** and **tracking success** information.

The format is:

```
scene,sequence,annotation,<object_path>,success,...
```

* **scene** – scene name (e.g., `2024-06-09`)
* **sequence** – sequence ID (3- or 4-digit)
* **\<object\_path>** – mesh file path (e.g., `0_tea/cirtus_reamer2/cirtus_reamer2.obj`)

Interpretation of fields:

* **Presence** → `"T"` under `<object_path>` if the object appears in that sequence.
* **Success** → `"T"` under `success` if the tracking result has been manually validated as correct.


## 📊 Tracking Statistics

The tracking pipeline is described in the supplementary material of our CVPR 2025 paper. It is based on **multi-view segmentation with differentiable rendering**.

Due to segmentation inconsistencies, tracking reliability cannot be measured purely by IoU. To ensure quality, **all tracking results have been manually annotated**.

* **Number of objects tracked**: 17,979
* **Number of sequences with successful tracking**: 3,356

We plan to further refine these results using latest models for segmentation. 

## 🎥 Example Use

After downloading hand pose, object pose and object meshes, run the script below to visualize hand-object mesh. 

[▶ Video demo (MP4, 211k)](visualizations/17_instruments/p003-instrument_0033/output.mp4)
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
<img src='visualizations/17_instruments/p003-instrument_0033/output.gif' width=480>