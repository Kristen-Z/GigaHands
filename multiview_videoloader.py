from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, Optional, Tuple

from video_utils.reader_cpu import Reader
from video_utils.video_handler import render_multiview
from video_utils.cameras import removed_cameras, map_camera_names, get_projections
import video_utils.params as param_utils
from video_utils.parser import add_common_args
import video_utils.tests as test_funcs

# ---------------------------
# Utilities / small helpers
# ---------------------------

def info(msg: str) -> None:
    print(f"[INFO] {msg}")

def warn(msg: str) -> None:
    print(f"[WARN] {msg}")

def err(msg: str) -> None:
    print(f"[ERROR] {msg}")

def load_params(params_path: Path):
    """Load camera/optimization parameters from a text file."""
    if not params_path.is_file():
        raise FileNotFoundError(f"Params file not found: {params_path}")
    return param_utils.read_params(str(params_path))

def load_ignored_cameras(ignore_path: Path) -> Optional[list[str]]:
    """Load ignore list from ignore_camera.txt (one camera name per line)."""
    if not ignore_path.is_file():
        return None
    with ignore_path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def select_cameras_to_remove(
    remove_side: bool,
    remove_side_bottom: bool,
    ignored_cameras: Optional[list[str]],
) -> list[str]:
    """Use your project’s policy to decide which camera names to remove."""
    return removed_cameras(
        remove_side=remove_side,
        remove_side_bottom=remove_side_bottom,
        ignored_cameras=ignored_cameras,
    )

def list_cameras_after_removal(all_cam_names: Iterable[str], cams_to_remove: set[str]) -> list[str]:
    """Return camera names preserved after removal, preserving the original order."""
    out = []
    for c in all_cam_names:
        if c not in cams_to_remove:
            out.append(c)
    return out

def find_anchor_camera_by_count(video_root: Path, cams_to_skip: set[str]) -> Tuple[str, int]:
    """
    Scan subfolders under session root, find the camera folder with the most .mp4 files.
    Returns (camera_folder_name, count).
    """
    best_cam = ""
    best_count = 0
    if not video_root.is_dir():
        warn(f"Video root does not exist: {video_root}")
        return best_cam, best_count

    for entry in video_root.iterdir():
        if not entry.is_dir():
            continue
        name = entry.name
        if "cam" not in name:
            continue
        if name in cams_to_skip:
            continue
        try:
            n_mp4 = sum(1 for f in entry.iterdir() if f.is_file() and f.suffix.lower() == ".mp4")
        except PermissionError:
            warn(f"No permission to read: {entry}")
            n_mp4 = 0
        if n_mp4 > best_count:
            best_count = n_mp4
            best_cam = name

    return best_cam, best_count

def compute_index_window(start: int, end: int, total: int) -> list[int]:
    """
    Build a list of available sequence indices within [start, end),
    clamped to [0, total).
    """
    s = max(start, 0)
    e = end if end > 0 else total
    e = min(e, total)
    if e < s:
        return []
    return list(range(s, e))


# ---------------------------
# Main orchestration
# ---------------------------

def run(args) -> None:
    # Resolve paths
    out_dir = Path(args.out_dir)
    session_root = Path(args.video_root_dir) / args.session
    params_path = out_dir / args.session / "optim_params.txt"
    ignore_path = out_dir / args.session / "ignore_camera.txt"

    # Read camera params & names
    params = load_params(params_path)
    params_cam_names = list(params[:]["cam_name"])
    info(f"Loaded params for {len(params_cam_names)} cameras.")

    # Ignored cameras file (optional)
    ignored_cameras = load_ignored_cameras(ignore_path)
    if ignored_cameras:
        info(f"Loaded {len(ignored_cameras)} ignored cameras from {ignore_path.name}.")
    else:
        info("No ignore_camera.txt found; skipping ignore list.")

    # Compute cameras to remove, then keep remainder
    cams_to_remove = set(
        select_cameras_to_remove(
            remove_side=args.remove_side_cam,
            remove_side_bottom=args.remove_side_bottom_cam,
            ignored_cameras=ignored_cameras,
        )
    )
    if cams_to_remove:
        info(f"To remove (policy): {sorted(cams_to_remove)}")
    kept_cam_names = list_cameras_after_removal(params_cam_names, cams_to_remove)
    info(f"{len(kept_cam_names)} cameras remain after filtering.")

    # Map capture camera names (as on disk) to params names (as in calibration)
    capture_cam_mapper = map_camera_names(str(session_root), kept_cam_names)
    capture_cam_names = sorted(n for n in kept_cam_names if n in set(capture_cam_mapper.keys()))
    if not capture_cam_names:
        warn("No overlapping camera names between params and capture folders after filtering.")

    # Pick anchor camera (by max .mp4 count), unless user provided one and seqid != -1
    auto_anchor, total_video_idxs = find_anchor_camera_by_count(session_root, cams_to_remove)
    anchor_camera = auto_anchor if args.seqid == -1 else args.anchor_camera
    info(f"Anchor camera: {anchor_camera or '(none)'}  |  Auto-suggested: {auto_anchor} ({total_video_idxs} mp4s)")

    # Determine sequence index window
    all_available = compute_index_window(args.start, args.end, total_video_idxs)
    info(f"Total sequences available in session (by anchor count): {len(all_available)}")
    info(f"Selected sequence id: {args.seqid}")

    # Build Reader
    reader = Reader(
        str(session_root),
        cams_to_remove=list(cams_to_remove),
        ith=args.seqid,
        anchor_camera=anchor_camera,
    )
    if reader.frame_count <= 0:
        err("Selected video is empty or index out of range. "
            "Check --seqid against available sequences.")
        raise SystemExit(1)

    # Post timestamp-alignment removal: update camera list actually used by reader
    reader_cams_to_remove = set(getattr(reader, "to_delete", []))
    reader_cam_names = list_cameras_after_removal(capture_cam_names, reader_cams_to_remove)

    # Calibrations/projections for surviving views
    intrs, projs, dists, cameras = get_projections(args, params, reader_cam_names)

    # Diagnostics
    info(f"Camera views (kept): {len(reader_cam_names)}")
    info(f"Intr/Proj/Dist lengths: {len(intrs)}, {len(projs)}, {len(dists)}")
    info(f"Reader views: {len(reader.vids)} | Total frames: {reader.frame_count}")

    # --- Tests / Demo Blocks ---
    info("--- Test 1: list views ---")
    test_funcs.test_list_views(reader)

    info("--- Test 2: load frames for a single view ---")
    if reader_cam_names:
        test_view_name = list(reader.streams.keys())[0]
        intr0, proj0, dist0 = intrs[0], projs[0], dists[0]
        _all_frames_np, _ts_list = test_funcs.test_load_all_frames_for_view(
            reader,
            test_view_name,
            undistort=True,
            intr=intr0,
            proj=proj0,
            dist=dist0,
        )
    else:
        warn("No reader_cam_names found after alignment; skipping Test 2.")

    info("--- Test 3: multiview frame batch ---")
    view_batches, ts_lists, view_names_list = [], [], []
    for view_names, frame_batch, ts_list in test_funcs.test_multiview_frame_batch(
        reader,
        frame_ids=[0, 1, 2],
        undistort=True,
        intrs=intrs,
        projs=projs,
        dists=dists,
    ):
        info(f"Views: {view_names} | Frame batch: {tuple(frame_batch.shape)}")
        view_batches.append(frame_batch)   # (V, H, W, 3)
        ts_lists.append(ts_list)
        view_names_list.append(view_names)

    render_multiview(
        view_batches=view_batches,
        view_names_list=view_names_list,
        output_dir=str(Path("visualizations") / "multview_output"),
    )

    info("Done.")


def build_parser() -> argparse.ArgumentParser:
    """
    Build CLI parser. We keep your original flags but switch to clearer bool handling.
    """
    parser = argparse.ArgumentParser(description="2D Keypoint / Multiview Driver")
    add_common_args(parser)

    # Better boolean flags: provide --no-* counterparts
    g = parser.add_argument_group("camera filtering")
    g.add_argument(
        "--remove-side-cam",
        dest="remove_side_cam",
        action="store_true",
        default=True,
        help="Remove Side Cameras that may have reflections (default: True).",
    )
    g.add_argument(
        "--keep-side-cam",
        dest="remove_side_cam",
        action="store_false",
        help="Keep Side Cameras (override).",
    )
    g.add_argument(
        "--remove-side-bottom-cam",
        dest="remove_side_bottom_cam",
        action="store_true",
        default=True,
        help="Remove Bottom Cameras that may have reflections (default: True).",
    )
    g.add_argument(
        "--keep-side-bottom-cam",
        dest="remove_side_bottom_cam",
        action="store_false",
        help="Keep Bottom Cameras (override).",
    )

    # In your add_common_args, you likely already have:
    #   --video_root_dir, --session, --out_dir, --seqid, --start, --end, --anchor_camera
    # If not, add them here.

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
