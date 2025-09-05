import os
import numpy as np
import video_utils.params as param_utils

SIDE_CAMERAS = ["brics-odroid-004_cam0", "brics-odroid-005_cam1"] # camera at the left and right of the hand panel
BOTTOM_SIDE_CAMERAS = [ # camera at the lower rows
                  "brics-odroid-003_cam0",
                  "brics-odroid-003_cam1",
                  "brics-odroid-004_cam0"
                  "brics-odroid-008_cam0",
                  "brics-odroid-008_cam1",
                  "brics-odroid-009_cam0",
                  "brics-odroid-013_cam0",
                  "brics-odroid-013_cam1",
                  "brics-odroid-014_cam0",
                  "brics-odroid-018_cam0",
                  "brics-odroid-018_cam1",
                  "brics-odroid-019_cam0",
                ]

BOTTOM_BOTTOM_CAMERAS = [ # camera at the bottom
                  "brics-odroid-026_cam0",
                  "brics-odroid-026_cam1",
                  "brics-odroid-027_cam0",
                  "brics-odroid-027_cam1",
                  "brics-odroid-028_cam0",
                  "brics-odroid-029_cam0",
                  "brics-odroid-029_cam1",
                  "brics-odroid-030_cam0",
                  "brics-odroid-030_cam1",
                ]

def removed_cameras(remove_side=True, remove_side_bottom=False, remove_bottom_bottom=False, ignored_cameras=None):
    
    if ignored_cameras:
        IGNORE_CAMERAS = ignored_cameras
    else:
        IGNORE_CAMERAS = []
    
    to_remove = IGNORE_CAMERAS
    if remove_side:
        to_remove += SIDE_CAMERAS
    if remove_side_bottom:
        to_remove += BOTTOM_SIDE_CAMERAS
    if remove_bottom_bottom:
        to_remove += BOTTOM_BOTTOM_CAMERAS
    return to_remove

def map_camera_names(base_dir, name_list):
    """
    Maps each name in the name_list to a subdirectory in base_dir that starts with the name.

    :param base_dir: The directory to search for subdirectories.
    :param name_list: A list of names to map to subdirectories.
    :return: A dictionary mapping each name in name_list to a matching subdirectory in base_dir.
    """
    # Find all subdirectories in the base directory
    subdirs = [d.split('.')[0] for d in os.listdir(base_dir)]

    # Create a dictionary to map names in name_list to subdirectories
    name_map = {}

    for name in name_list:
        # Find a subdirectory that starts with the name
        matched_subdir = next((subdir for subdir in subdirs if subdir.startswith(name)), None)
        
        if matched_subdir:
            name_map[name] = matched_subdir

    return name_map

def get_projections(args, params, cam_names):
    # Gets the projection matrices and distortion parameters
    filtered_cam_names = []
    projs = []
    intrs = []
    dists = []
    rot = []
    trans = []

    for param in params:
        if param["cam_name"] in cam_names:
            extr = param_utils.get_extr(param)
            intr, dist = param_utils.get_intr(param)
            r, t = param_utils.get_rot_trans(param)

            rot.append(r)
            trans.append(t)

            intrs.append(intr.copy())
            projs.append(intr @ extr)
            dists.append(dist)
            filtered_cam_names.append(param["cam_name"])

    cameras = { 
        'cam_names': filtered_cam_names,
        'K': np.asarray(intrs),
        'R': np.asarray(rot), 
        'T': np.asarray(trans),
        'dist': np.asarray(dists),
        'P': np.asarray(projs) 
    }
    
    return intrs, np.asarray(projs), dists, cameras