#!/usr/bin/env python3
"""Collect HORA ShadowHand v4 tennis-ball rollouts with the WiLoR view.

This collector mirrors the validated DexterousHands raw-rigid-contact path:
RGB frames, robot/object state, and EgoTouch-layout pressure grids are recorded
from the same simulation steps.  HORA v4 is a single-hand ShadowHand task, so
the right-hand grid contains the projected contacts from actor ``hand`` and the
left-hand grid is an empty canonical layout.
"""

import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import isaacgym  # noqa: F401  Must precede torch and Isaac Gym submodule imports.
import hydra
import imageio.v2 as imageio
import numpy as np
import torch
from isaacgym import gymapi
from omegaconf import DictConfig, OmegaConf

from hora.algo.ppo.ppo import PPO
from hora.tasks import isaacgym_task_map
from hora.utils.reformat import omegaconf_to_dict


OmegaConf.register_new_resolver("eq", lambda x, y: x.lower() == y.lower())
OmegaConf.register_new_resolver("contains", lambda x, y: x.lower() in y.lower())
OmegaConf.register_new_resolver("if", lambda p, a, b: a if p else b)
OmegaConf.register_new_resolver("resolve_default", lambda d, a: d if a == "" else a)


def _prepend_dexteroushands_path():
    dex_root = os.environ.get("DEXTEROUSHANDS_ROOT")
    candidates = [dex_root] if dex_root else []
    candidates += ["/lp-dev/qianqian/DexterousHands", "/workspace/DexterousHands", "/workspace"]
    for root in candidates:
        if not root:
            continue
        bidex = Path(root) / "bidexhands"
        mapper = bidex / "tactile_collection" / "egotouch_taxels.py"
        if mapper.exists():
            sys.path.insert(0, str(bidex))
            return Path(root), bidex
    raise RuntimeError("Cannot locate DexterousHands/bidexhands for EgoTouch taxel mapper")


DEX_ROOT, BIDEX_ROOT = _prepend_dexteroushands_path()
from tactile_collection.egotouch_taxels import EgoTouchTaxelMapper  # noqa: E402


def as_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def parse_vec3_env(name, default):
    values = [float(x.strip()) for x in os.environ.get(name, default).split(",")]
    if len(values) != 3:
        raise ValueError("{} must have 3 comma-separated floats".format(name))
    return np.asarray(values, dtype=np.float32)


def parse_bool_env(name, default=False):
    text = os.environ.get(name)
    if text is None:
        return bool(default)
    return text.strip().lower() in ("1", "true", "yes", "y", "on")


def scalar_text(value):
    return np.asarray(value, dtype=np.str_)


def set_visual_style(env):
    hand_color = parse_vec3_env("HORA_HAND_COLOR_RGB", "0.42,0.52,0.56")
    obj_color = parse_vec3_env("HORA_OBJECT_COLOR_RGB", "0.40,0.58,0.28")
    for actor_name, color in (("hand", hand_color), ("object", obj_color)):
        actor = env.gym.find_actor_handle(env.envs[0], actor_name)
        if actor < 0:
            continue
        body_count = env.gym.get_actor_rigid_body_count(env.envs[0], actor)
        for body_id in range(body_count):
            env.gym.set_rigid_body_color(
                env.envs[0], actor, body_id, gymapi.MESH_VISUAL, gymapi.Vec3(*color.tolist())
            )


def body_env_indices(env, actor_name, name_patterns=None):
    actor = env.gym.find_actor_handle(env.envs[0], actor_name)
    if actor < 0:
        return []
    names = env.gym.get_actor_rigid_body_names(env.envs[0], actor)
    patterns = tuple(p.lower() for p in name_patterns) if name_patterns else None
    indices = []
    for local_index, name in enumerate(names):
        if patterns and not any(pattern in name.lower() for pattern in patterns):
            continue
        indices.append(
            env.gym.get_actor_rigid_body_index(env.envs[0], actor, local_index, gymapi.DOMAIN_ENV)
        )
    return indices


def positions_from_body_indices(env, body_indices):
    if not body_indices:
        return None
    states = as_numpy(env.rigid_body_states)[0]
    valid = [int(i) for i in body_indices if 0 <= int(i) < states.shape[0]]
    if not valid:
        return None
    return states[np.asarray(valid, dtype=np.int64), :3].astype(np.float32)


def workspace_center(env):
    points = []
    tactile_names = ("palm", "distal", "middle", "proximal", "metacarpal")
    hand_pts = positions_from_body_indices(env, body_env_indices(env, "hand", tactile_names))
    obj_pts = positions_from_body_indices(env, body_env_indices(env, "object"))
    if hand_pts is not None:
        points.append(hand_pts)
    if obj_pts is not None:
        points.append(obj_pts)
    if not points:
        return as_numpy(env.object_pos)[0].astype(np.float32)
    pts = np.concatenate(points, axis=0)
    return ((pts.min(axis=0) + pts.max(axis=0)) * 0.5).astype(np.float32)


def create_camera(env, width, height):
    props = gymapi.CameraProperties()
    props.width = width
    props.height = height
    props.enable_tensors = False
    props.horizontal_fov = float(os.environ.get("HORA_CAMERA_FOV", "50.0"))
    camera = env.gym.create_camera_sensor(env.envs[0], props)
    if camera < 0:
        raise RuntimeError("Isaac Gym failed to create camera")
    return camera


def position_camera(env, camera):
    target = workspace_center(env) + parse_vec3_env("HORA_CHEST_TARGET_OFFSET", "0.0,0.0,0.08")
    eye = workspace_center(env) + parse_vec3_env("HORA_CHEST_EYE_OFFSET", "0.32,0.0,0.80")
    env.gym.set_camera_location(
        camera, env.envs[0], gymapi.Vec3(*eye.tolist()), gymapi.Vec3(*target.tolist())
    )
    return eye, target


def capture_frame(env, camera, width, height, path):
    env.gym.fetch_results(env.sim, True)
    env.gym.step_graphics(env.sim)
    env.gym.render_all_camera_sensors(env.sim)
    rgba = np.asarray(
        env.gym.get_camera_image(env.sim, env.envs[0], camera, gymapi.IMAGE_COLOR),
        dtype=np.uint8,
    ).reshape(height, width, 4)
    imageio.imwrite(path, rgba[:, :, :3].copy())


def encode_rgb(frames_dir, output_mp4, fps):
    subprocess.check_call([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
        "-i", str(Path(frames_dir) / "frame_%06d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(output_mp4),
    ])


def render_tactile(pressure_path, output_mp4, fps, stride):
    script = BIDEX_ROOT / "tactile_collection" / "render_tactile.py"
    subprocess.check_call([
        os.environ.get("PYTHON", sys.executable), str(script), str(pressure_path), str(output_mp4),
        "--fps", str(fps), "--stride", str(stride),
    ])


def compose_side_by_side(rgb_mp4, tactile_mp4, output_mp4):
    subprocess.check_call([
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(rgb_mp4), "-i", str(tactile_mp4),
        "-filter_complex",
        "[0:v]scale=800:-2,pad=800:600:(ow-iw)/2:(oh-ih)/2,setsar=1[rgb];"
        "[1:v]scale=1200:600,setsar=1[tactile];"
        "[rgb][tactile]hstack=inputs=2[v]",
        "-map", "[v]", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
        str(output_mp4),
    ])


def quat_to_axis_angle_np(q):
    q = np.asarray(q, dtype=np.float64)
    if q[3] < 0.0:
        q = -q
    w = np.clip(q[3], -1.0, 1.0)
    angle = 2.0 * math.acos(w)
    s = math.sqrt(max(1.0 - w * w, 0.0))
    if s < 1.0e-8:
        return np.zeros(3, dtype=np.float64)
    return q[:3] / s * angle


def quat_mul_np(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.asarray([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], dtype=np.float64)


def quat_conj_np(q):
    return np.asarray([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)


def episode_quality(episode):
    steps = len(episode["right_pressure_grid"])
    object_z = np.asarray(episode["object_pos"], dtype=np.float32)[:, 2] if episode["object_pos"] else np.asarray([])
    contact_count = np.asarray(episode["right_contact_count"], dtype=np.int32)
    cumulative = np.asarray(episode["cumulative_rotation_z_rad"], dtype=np.float32)
    drop_threshold = float(os.environ.get("HORA_DROP_Z_THRESHOLD", "0.48"))
    max_rot = float(np.max(np.abs(cumulative))) if cumulative.size else 0.0
    dropped = bool(object_z.min() < drop_threshold) if object_z.size else None
    return {
        "steps": int(steps),
        "rgb_frames": int(len(episode["frames"])),
        "total_contacts": int(contact_count.sum()) if contact_count.size else 0,
        "max_cumulative_rotation_abs_rad": max_rot,
        "full_2pi_rotation": bool(max_rot >= 2.0 * math.pi),
        "terminated": bool(episode["done"][-1]) if episode["done"] else False,
        "drop_z_threshold_m": drop_threshold,
        "min_object_z_m": float(object_z.min()) if object_z.size else None,
        "dropped": dropped,
        "keep_success": bool(max_rot >= 2.0 * math.pi and dropped is False and contact_count.sum() > 0),
    }


def should_write_qa_video(episode_idx, saved_count):
    if not parse_bool_env("HORA_WRITE_QA_VIDEO", True):
        return False
    every = max(1, int(os.environ.get("HORA_QA_VIDEO_EVERY", "1")))
    limit = int(os.environ.get("HORA_QA_VIDEO_LIMIT", "0"))
    if episode_idx % every != 0:
        return False
    return limit <= 0 or saved_count < limit


def save_episode_artifact(root, episode_idx, episode, right_mapper, frame_stride, fps):
    ep_dir = root / "episodes" / "episode_{:06d}".format(episode_idx)
    frame_dir = ep_dir / "rgb_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    for i, frame in enumerate(episode["frames"]):
        imageio.imwrite(frame_dir / "frame_{:06d}.png".format(i), frame)

    steps = len(episode["right_pressure_grid"])
    valid = right_mapper.valid_mask
    left_grid = np.full((steps, 21, 21), np.nan, dtype=np.float32)
    left_force_grid = np.zeros((steps, 21, 21), dtype=np.float32)
    left_grid[:, valid] = 0.0
    pressure = {
        "left_pressure_grid": left_grid,
        "right_pressure_grid": np.asarray(episode["right_pressure_grid"], dtype=np.float32),
        "left_force_grid_n": left_force_grid,
        "right_force_grid_n": np.asarray(episode["right_force_grid_n"], dtype=np.float32),
        "left_source_force_n": np.zeros(steps, dtype=np.float32),
        "right_source_force_n": np.asarray(episode["right_source_force_n"], dtype=np.float32),
        "left_reconstructed_force_n": np.zeros(steps, dtype=np.float32),
        "right_reconstructed_force_n": np.asarray(episode["right_reconstructed_force_n"], dtype=np.float32),
        "left_contact_count": np.zeros(steps, dtype=np.int32),
        "right_contact_count": np.asarray(episode["right_contact_count"], dtype=np.int32),
        "left_valid_mask": valid,
        "right_valid_mask": valid,
        "left_taxel_area_m2": right_mapper.taxel_area_m2.astype(np.float32),
        "right_taxel_area_m2": right_mapper.taxel_area_m2.astype(np.float32),
        "pressure_unit": scalar_text("Pa"),
        "force_unit": scalar_text("N"),
        "area_unit": scalar_text("m^2"),
        "normalization": scalar_text("none"),
        "layout": scalar_text("EgoTouch-21x21-217-taxels-per-hand"),
        "grid_size": np.asarray(21, dtype=np.int32),
        "num_frames": np.asarray(steps, dtype=np.int32),
        "frame_index": np.asarray(episode["frame_index"], dtype=np.int32),
        "video_frame_stride": np.asarray(frame_stride, dtype=np.int32),
        "video_fps": np.asarray(fps, dtype=np.int32),
        "pressure_definition": scalar_text(
            "allocated Isaac RigidContact.lambda normal force [N] / represented taxel area [m^2]"
        ),
        "single_hand_note": scalar_text("HORA ShadowHand v4 has one actor named hand; contacts are stored in right_* grids."),
    }
    pressure_path = ep_dir / "pressure_grids.npz"
    np.savez_compressed(pressure_path, **pressure)

    trace = {
        key: np.asarray(value)
        for key, value in episode.items()
        if key not in ("frames", "right_pressure_grid", "right_force_grid_n")
    }
    trace["single_hand_note"] = scalar_text("HORA ShadowHand v4 one-hand rollout; pressure_grids right_* is actor hand.")
    trace_path = ep_dir / "trajectory_env0.npz"
    np.savez_compressed(trace_path, **trace)

    rgb_mp4 = ep_dir / "rgb.mp4"
    tactile_mp4 = ep_dir / "tactile.mp4"
    paired_mp4 = ep_dir / "rgb_tactile_side_by_side.mp4"
    write_video = bool(episode.get("_write_qa_video", True))
    if write_video:
        encode_rgb(frame_dir, rgb_mp4, fps)
        render_tactile(pressure_path, tactile_mp4, fps, frame_stride)
        compose_side_by_side(rgb_mp4, tactile_mp4, paired_mp4)
        if not parse_bool_env("HORA_KEEP_COMPONENT_VIDEOS", False):
            for component in (rgb_mp4, tactile_mp4):
                if component.exists():
                    component.unlink()
    if not parse_bool_env("HORA_KEEP_RGB_FRAMES", True):
        shutil.rmtree(frame_dir)

    valid_pressures = pressure["right_pressure_grid"][:, valid]
    quality = episode_quality(episode)
    artifact = {
        "episode_id": int(episode_idx),
        "steps": quality["steps"],
        "rgb_frames": quality["rgb_frames"],
        "pressure_grids": str(pressure_path),
        "trajectory": str(trace_path),
        "rgb_frames_dir": str(frame_dir) if frame_dir.exists() else None,
        "side_by_side_video": str(paired_mp4) if write_video else None,
        "qa_video_written": bool(write_video),
        "max_pressure_pa": float(np.nanmax(valid_pressures)) if valid_pressures.size else 0.0,
        "nonzero_pressure_fraction": float(np.nanmean(valid_pressures > 0)) if valid_pressures.size else 0.0,
    }
    artifact.update(quality)
    with open(ep_dir / "episode_manifest.json", "w") as handle:
        json.dump(artifact, handle, indent=2, sort_keys=True)
    return artifact


def write_run_manifest(out, artifacts, cfg, right_mapper, width, height, fps, frame_stride, start, total_steps, final=False):
    max_pressure = max((a["max_pressure_pa"] for a in artifacts), default=0.0)
    manifest = {
        "name": "hora_shadow_v4_tennis_wilor_view_raw_rigid",
        "task": cfg.task_name,
        "checkpoint": str(Path(cfg.checkpoint).resolve()),
        "output": str(out),
        "elapsed_seconds": time.time() - start,
        "total_sim_steps": int(total_steps),
        "episodes": len(artifacts),
        "artifacts": artifacts,
        "object_type": cfg.task.env.object.type,
        "grasp_cache_name": cfg.task.env.grasp_cache_name,
        "policy_output_name": cfg.train.ppo.output_name,
        "camera": {
            "mode": "wilor_chest_workspace",
            "eye_offset_m": [float(x) for x in parse_vec3_env("HORA_CHEST_EYE_OFFSET", "0.32,0.0,0.80")],
            "target_offset_m": [float(x) for x in parse_vec3_env("HORA_CHEST_TARGET_OFFSET", "0.0,0.0,0.08")],
            "width": width,
            "height": height,
            "fps": fps,
            "frame_stride": frame_stride,
        },
        "visual_style": {
            "hand_color_rgb": os.environ.get("HORA_HAND_COLOR_RGB", "0.42,0.52,0.56"),
            "object_color_rgb": os.environ.get("HORA_OBJECT_COLOR_RGB", "0.40,0.58,0.28"),
        },
        "tactile": {
            "contact_projection": "rigid_contacts",
            "normalization": "none",
            "layout": "EgoTouch 21x21; right grid = HORA actor hand; left grid empty",
            "pressure_unit": "Pa",
            "max_pressure_pa": max_pressure,
            "right_mapper": right_mapper.metadata(),
        },
    }
    with open(out / "shard_manifest.json", "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    if final:
        with open(out / "summary.json", "w") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
    return manifest


@hydra.main(config_name="config", config_path="configs")
def main(cfg: DictConfig):
    if cfg.checkpoint:
        from hydra.utils import to_absolute_path
        cfg.checkpoint = to_absolute_path(cfg.checkpoint)
    task_cfg = omegaconf_to_dict(cfg.task)
    task_cfg["enableCameraSensors"] = True
    env = isaacgym_task_map[cfg.task_name](task_cfg, cfg.sim_device, cfg.graphics_device_id, headless=cfg.headless)
    set_visual_style(env)
    agent = PPO(env, "outputs/hora_wilor_collect_tmp", full_config=cfg)
    agent.restore_test(cfg.train.load_path)
    agent.set_eval()

    out = Path(os.environ.get("HORA_TACTILE_DIR", "/lp-dev/qianqian/hora/outputs/ShadowHandHora/v4_spread_wilor_raw_rigid"))
    if (out / "summary.json").exists() and not parse_bool_env("ALLOW_EXISTING_OUTPUT", False):
        raise RuntimeError("Refusing to overwrite existing output {}".format(out))
    out.mkdir(parents=True, exist_ok=True)

    width = int(os.environ.get("HORA_VIDEO_WIDTH", "960"))
    height = int(os.environ.get("HORA_VIDEO_HEIGHT", "720"))
    fps = int(os.environ.get("HORA_VIDEO_FPS", "30"))
    frame_stride = max(1, int(os.environ.get("HORA_FRAME_STRIDE", "2")))
    max_steps = int(os.environ.get("HORA_MAX_STEPS", "900"))
    target_episodes = int(os.environ.get("HORA_TARGET_EPISODES", "2"))
    warmup = int(os.environ.get("HORA_WARMUP_STEPS", "0"))
    keep_only_success = parse_bool_env("HORA_KEEP_ONLY_SUCCESS", False)

    asset_dir = BIDEX_ROOT / "tactile_collection" / "assets"
    right_mapper = EgoTouchTaxelMapper(
        env.gym, env.envs[0], "hand", "right", str(asset_dir / "pressure_position_mapping_right.json")
    )
    camera = create_camera(env, width, height)

    object_actor = env.gym.find_actor_handle(env.envs[0], "object")
    object_body_names = env.gym.get_actor_rigid_body_names(env.envs[0], object_actor)
    hand_actor = env.gym.find_actor_handle(env.envs[0], "hand")
    hand_body_names = env.gym.get_actor_rigid_body_names(env.envs[0], hand_actor)
    hand_dof_names = env.gym.get_actor_dof_names(env.envs[0], hand_actor)

    obs = env.reset()
    artifacts = []
    current = None
    prev_rot = as_numpy(env.object_rot)[0].copy()
    cumulative = 0.0
    start = time.time()
    total_steps = 0

    for step in range(max_steps):
        with torch.no_grad():
            inp = {"obs": agent.running_mean_std(obs["obs"]), "priv_info": obs.get("priv_info", None)}
            action = torch.clamp(agent.model.act_inference(inp), -1.0, 1.0)
            obs, rew, done, info = env.step(action)
        total_steps += 1
        if step < warmup:
            prev_rot = as_numpy(env.object_rot)[0].copy()
            continue
        if current is None:
            current = {
                "frame_index": [], "reward": [], "done": [], "actions": [],
                "shadow_hand_dof_pos": [], "object_pose": [], "object_pos": [],
                "object_rot": [], "object_rigid_body_state": [], "hand_rigid_body_state": [],
                "rgb_frame_step": [], "rgb_camera_eye": [], "rgb_camera_target": [],
                "right_pressure_grid": [], "right_force_grid_n": [],
                "right_source_force_n": [], "right_reconstructed_force_n": [],
                "right_contact_count": [], "cumulative_rotation_z_rad": [],
                "object_angvel": [], "frames": [],
            }
            cumulative = 0.0
            prev_rot = as_numpy(env.object_rot)[0].copy()

        contacts = env.gym.get_env_rigid_contacts(env.envs[0])
        right_pa, right_force, right_diag = right_mapper.project(contacts)
        current["right_pressure_grid"].append(right_pa)
        current["right_force_grid_n"].append(right_force)
        current["right_source_force_n"].append(right_diag["source_force_n"])
        current["right_reconstructed_force_n"].append(right_diag["reconstructed_force_n"])
        current["right_contact_count"].append(right_diag["contact_count"])

        obj_rot = as_numpy(env.object_rot)[0].copy()
        rel = quat_mul_np(obj_rot, quat_conj_np(prev_rot))
        delta_z = float(quat_to_axis_angle_np(rel)[2])
        cumulative += delta_z
        prev_rot = obj_rot

        current["frame_index"].append(step)
        current["reward"].append(float(as_numpy(rew)[0]))
        current["done"].append(bool(as_numpy(done)[0]))
        current["actions"].append(as_numpy(action)[0].copy())
        current["shadow_hand_dof_pos"].append(as_numpy(env.allegro_hand_dof_pos)[0].copy())
        current["object_pose"].append(as_numpy(env.object_pose)[0].copy())
        current["object_pos"].append(as_numpy(env.object_pos)[0].copy())
        current["object_rot"].append(obj_rot.copy())
        current["object_rigid_body_state"].append(as_numpy(env.rigid_body_states)[0, env.num_allegro_hand_bodies].copy())
        current["hand_rigid_body_state"].append(as_numpy(env.rigid_body_states)[0, :env.num_allegro_hand_bodies].copy())
        current["cumulative_rotation_z_rad"].append(cumulative)
        current["object_angvel"].append(as_numpy(env.rigid_body_states)[0, env.num_allegro_hand_bodies, 10:13].copy())

        if len(current["frame_index"]) % frame_stride == 1:
            eye, target = position_camera(env, camera)
            tmp_path = out / "_frame_tmp.png"
            capture_frame(env, camera, width, height, tmp_path)
            current["frames"].append(imageio.imread(tmp_path))
            if tmp_path.exists():
                tmp_path.unlink()
            current["rgb_frame_step"].append(step)
            current["rgb_camera_eye"].append(eye)
            current["rgb_camera_target"].append(target)

        if bool(as_numpy(done)[0]):
            quality = episode_quality(current)
            print("EPISODE_DONE idx={} steps={} max_abs_rot={:.3f} contacts={}".format(
                len(artifacts),
                quality["steps"],
                quality["max_cumulative_rotation_abs_rad"],
                quality["total_contacts"],
            ), flush=True)
            if keep_only_success and not quality["keep_success"]:
                print("EPISODE_SKIP idx={} reason=quality_failed {}".format(
                    len(artifacts), json.dumps(quality, sort_keys=True)
                ), flush=True)
            else:
                current["_write_qa_video"] = should_write_qa_video(len(artifacts), sum(a["qa_video_written"] for a in artifacts))
                artifact = save_episode_artifact(out, len(artifacts), current, right_mapper, frame_stride, fps)
                artifacts.append(artifact)
                write_run_manifest(out, artifacts, cfg, right_mapper, width, height, fps, frame_stride, start, total_steps)
                print("EPISODE_SAVED idx={} keep_success={} qa_video={} dir={}".format(
                    artifact["episode_id"], artifact["keep_success"], artifact["qa_video_written"],
                    Path(artifact["trajectory"]).parent,
                ), flush=True)
            current = None
            if len(artifacts) >= target_episodes:
                break

    if current is not None and len(artifacts) < target_episodes and parse_bool_env("HORA_SAVE_PARTIAL_EPISODE", False):
        quality = episode_quality(current)
        if not keep_only_success or quality["keep_success"]:
            current["_write_qa_video"] = should_write_qa_video(len(artifacts), sum(a["qa_video_written"] for a in artifacts))
            artifacts.append(save_episode_artifact(out, len(artifacts), current, right_mapper, frame_stride, fps))

    summary = write_run_manifest(out, artifacts, cfg, right_mapper, width, height, fps, frame_stride, start, total_steps, final=True)
    summary["hand_body_names"] = list(hand_body_names)
    summary["hand_dof_names"] = list(hand_dof_names)
    summary["object_body_names"] = list(object_body_names)
    with open(out / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print("HORA_WILOR_RAW_RIGID_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
