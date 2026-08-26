# Data-collection rollout for the pretrained Hora policy: 3rd-person + ego RGB video + per-hand-body
# contact force (tactile, "16-FSR"-style) + proprioception. Place at hora/ root (hydra config_path='configs').
import isaacgym  # noqa (must be before torch)
import json, os, numpy as np, torch, hydra
from omegaconf import DictConfig, OmegaConf
from isaacgym import gymapi
from hora.tasks import isaacgym_task_map
from hora.utils.reformat import omegaconf_to_dict
from hora.algo.ppo.ppo import PPO
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib import cm
import imageio.v2 as imageio

OmegaConf.register_new_resolver('eq', lambda x, y: x.lower() == y.lower())
OmegaConf.register_new_resolver('contains', lambda x, y: x.lower() in y.lower())
OmegaConf.register_new_resolver('if', lambda p, a, b: a if p else b)
OmegaConf.register_new_resolver('resolve_default', lambda d, a: d if a == '' else a)

N_STEPS = int(os.environ.get('COLLECT_STEPS', '400'))
OUT = os.environ.get('COLLECT_OUT', '/lp-dev/qianqian/hora_collect')


@hydra.main(config_name='config', config_path='configs')
def main(cfg: DictConfig):
    if cfg.checkpoint:
        from hydra.utils import to_absolute_path
        cfg.checkpoint = to_absolute_path(cfg.checkpoint)
    # force camera rendering ON (needed for offscreen cameras even when headless)
    task_cfg = omegaconf_to_dict(cfg.task)
    task_cfg['enableCameraSensors'] = True
    env = isaacgym_task_map[cfg.task_name](task_cfg, cfg.sim_device, cfg.graphics_device_id, cfg.headless)
    agent = PPO(env, 'outputs/collect_tmp', full_config=cfg)
    agent.restore_test(cfg.train.load_path)
    agent.set_eval()

    gym, sim = env.gym, env.sim
    obj_bi0 = env.num_allegro_hand_bodies
    # WARMUP: run the policy, then render the BEST-rotating env (env 0 is often below average)
    _obs = env.reset()
    _acc = torch.zeros(env.num_envs, device=env.device)
    for _t in range(int(os.environ.get('WARMUP', '60'))):
        _inp = {'obs': agent.running_mean_std(_obs['obs']), 'priv_info': _obs.get('priv_info', None)}
        _mu = torch.clamp(agent.model.act_inference(_inp), -1.0, 1.0)
        _obs, _, _, _ = env.step(_mu)
        _acc += env.rigid_body_states[:, obj_bi0, 12].abs()
    # NOTE: rendering a non-zero env mis-frames the camera (env grid-offset), so render env 0 (frames correctly).
    EI = int(os.environ.get('RENDER_ENV', '0'))
    print('rendering env:', EI, '| its warmup |wz|-accum:', round(float(_acc[EI]), 2), flush=True)
    e0 = env.envs[EI]
    hand = gym.find_actor_handle(e0, 'hand')
    _obj = gym.find_actor_handle(e0, 'object')
    # Rendering-only asymmetric texture.  A uniformly colored cube is
    # invariant under 90-degree yaw and makes real z-rotation look static.
    # This changes neither collision geometry nor task/observation state.
    texture_path = os.environ.get(
        'OBJ_TEXTURE',
        '/lp-dev/qianqian/hora/assets/open_ai_assets/textures/block.png')
    if texture_path and os.path.exists(texture_path):
        texture = gym.create_texture_from_file(sim, texture_path)
        gym.set_rigid_body_texture(
            e0, _obj, 0, gymapi.MESH_VISUAL, texture)
        print('visual-only object texture:', texture_path, flush=True)
    # keep the object's own texture (e.g. tennis ball) unless COLOR_OBJ=1; color fingers distinctly for gaiting
    if os.environ.get('COLOR_OBJ', '0') == '1':
        gym.set_rigid_body_color(e0, _obj, 0, gymapi.MESH_VISUAL, gymapi.Vec3(1.0, 0.5, 0.0))
    _fc = [(0.9,0.1,0.1),(0.1,0.8,0.1),(0.1,0.3,1.0),(0.95,0.85,0.1)]
    _nb = gym.get_actor_rigid_body_count(e0, hand)
    for _bi in range(_nb):
        c = gymapi.Vec3(0.4,0.4,0.4) if _bi == 0 else gymapi.Vec3(*_fc[(_bi-1)//4 % 4])
        gym.set_rigid_body_color(e0, hand, _bi, gymapi.MESH_VISUAL, c)
    body_names = gym.get_actor_rigid_body_names(e0, hand)
    n_hand = len(body_names)
    print('HAND BODIES (%d):' % n_hand, body_names, flush=True)
    # ego camera anchor = a palm/base body; else body 0
    anchor = 0
    # Prefer the actual palm.  The previous one-pass substring search selected
    # `wrist` first because it appears earlier in this asset's body list.
    for preferred in ['palm', 'wrist', 'base', 'link_0']:
        matches = [i for i, nm in enumerate(body_names)
                   if preferred in nm.lower()]
        if matches:
            anchor = matches[0]
            break
    anchor_handle = gym.get_actor_rigid_body_handle(e0, hand, anchor)
    print('ego anchor body:', body_names[anchor], flush=True)

    # --- cameras --- object body index in rigid_body_states = right after the hand bodies
    obj_bi = n_hand
    cp = gymapi.CameraProperties(); cp.width = 480; cp.height = 480; cp.enable_tensors = False
    hand_state = env.rigid_body_states[EI, anchor, :3].cpu().numpy()
    cx, cy, cz = float(hand_state[0]), float(hand_state[1]), float(hand_state[2])
    cam3 = gym.create_camera_sensor(e0, cp)     # HORA's EXACT default viewer camera (env line 53-54)
    _e = [float(v) for v in os.environ.get('CAM3_EYE', '0.55,0.55,0.9').split(',')]
    _t = [float(v) for v in os.environ.get('CAM3_TGT', '0.0,0.0,0.52').split(',')]
    gym.set_camera_location(cam3, e0, gymapi.Vec3(*_e), gymapi.Vec3(*_t))
    cam_ego = gym.create_camera_sensor(e0, cp)  # ego close-up: re-aimed at the object each step (set in loop)

    def obj_pose():
        s = env.rigid_body_states[EI, obj_bi].cpu().numpy()
        return s[:3], s[3:7]   # pos, quat(x,y,z,w)

    def quat_R(q):
        x, y, z, w = q
        return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                         [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                         [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

    def project(cam, wp):   # world point -> (u,v) pixel in that camera; IsaacGym row-vector convention
        V = np.array(gym.get_camera_view_matrix(sim, e0, cam))
        P = np.array(gym.get_camera_proj_matrix(sim, e0, cam))
        clip = np.array([wp[0], wp[1], wp[2], 1.0]) @ V @ P
        if clip[3] <= 1e-6:
            return None
        ndc = clip[:3] / clip[3]
        return ((ndc[0]*0.5+0.5)*cp.width, (1-(ndc[1]*0.5+0.5))*cp.height)

    # two markers painted on the cylinder SURFACE (opposite sides) so its rotation is visible as they orbit
    MARK_LOCAL = [np.array([0.038, 0.0, 0.0]), np.array([-0.038, 0.0, 0.0])]
    MARK_COL = ['red', 'yellow']

    # tactile grid layout (Allegro: group bodies by finger from their names for a readable panel)
    def finger_of(nm):
        n = nm.lower()
        for k, lab in [('0', 'index'), ('1', 'middle'), ('2', 'ring'), ('3', 'thumb')]:
            if f'_{k}_' in n or n.endswith('_' + k): return lab
        return 'palm'
    labels = [f'{i}:{body_names[i][:14]}' for i in range(n_hand)]

    os.makedirs(OUT, exist_ok=True)
    obs = _obs   # continue from warmup (keep the chosen env's rotation going)
    frames, tactile_log, qpos_log, action_log = [], [], [], []
    objrot_log, objpos_log, handpos_log, objquat_log = [], [], [], []   # ang-vel-z + positions + yaw to VERIFY rotation
    objquat_full_log, signed_delta_log, task_signed_delta_log, done_log = [], [], [], []
    dofpos_log = []   # the 16 Allegro JOINT ANGLES each step (to prove the fingers actually bend)
    fmax = 1e-6
    prev_object_quat = obj_pose()[1].copy()

    def physical_minus_z_delta(prev_q, curr_q):
        """Independent short-arc relative-quaternion rotation about world -z."""
        px, py, pz, pw = prev_q
        cx, cy, cz, cw = curr_q
        # curr * conjugate(prev), quaternion layout xyzw.
        q = np.array([
            cw * (-px) + cx * pw + cy * (-pz) - cz * (-py),
            cw * (-py) - cx * (-pz) + cy * pw + cz * (-px),
            cw * (-pz) + cx * (-py) - cy * (-px) + cz * pw,
            cw * pw - cx * (-px) - cy * (-py) - cz * (-pz),
        ], dtype=np.float64)
        if q[3] < 0.0:
            q = -q
        vn = float(np.linalg.norm(q[:3]))
        if vn < 1.0e-9:
            return 0.0
        angle = 2.0 * np.arctan2(vn, np.clip(q[3], -1.0, 1.0))
        return float(-(q[2] / vn) * angle)

    for t in range(N_STEPS):
        inp = {'obs': agent.running_mean_std(obs['obs']), 'priv_info': obs.get('priv_info', None)}
        mu = torch.clamp(agent.model.act_inference(inp), -1.0, 1.0)
        obs, r, done, info = env.step(mu)
        task_signed_delta_log.append(float(info['signed_delta_rad']))
        done_log.append(bool(done[EI]))
        # True palm-following optical center, dynamically aimed at the object.
        # Estimate the palm-face normal from the live fingertip geometry.  This
        # avoids placing the camera inside/behind the palm when the object is
        # almost directly above it (where a palm->object vector is near-z).
        o, oq = obj_pose()
        signed_delta_log.append(physical_minus_z_delta(prev_object_quat, oq))
        prev_object_quat = oq.copy()
        hand_xyz = env.rigid_body_states[EI, :n_hand, :3].cpu().numpy()
        palm_xyz = hand_xyz[anchor]
        tip_ids = [i for i, nm in enumerate(body_names)
                   if nm.lower().endswith('distal')]
        tips = hand_xyz[tip_ids]
        finger_axis = tips.mean(axis=0) - palm_xyz
        finger_axis /= max(float(np.linalg.norm(finger_axis)), 1.0e-6)
        spread = tips[-2] - tips[0] if len(tips) >= 2 else np.array([1.0, 0.0, 0.0])
        palm_normal = np.cross(finger_axis, spread)
        palm_normal /= max(float(np.linalg.norm(palm_normal)), 1.0e-6)
        if float(np.dot(palm_normal, o - palm_xyz)) < 0.0:
            palm_normal = -palm_normal
        eye = palm_xyz + 0.22 * palm_normal - 0.06 * finger_axis
        cluster_target = 0.65 * o + 0.35 * palm_xyz
        eye3 = o + np.array([0.28, 0.28, 0.18])
        gym.set_camera_location(cam3, e0, gymapi.Vec3(*eye3.tolist()),
                                gymapi.Vec3(*cluster_target.tolist()))
        gym.set_camera_location(cam_ego, e0, gymapi.Vec3(*eye.tolist()),
                                gymapi.Vec3(*cluster_target.tolist()))
        # Headless VecTask.render() does not advance the graphics scene.  The
        # physics/tensor state can therefore move while camera RGB remains
        # frozen.  Match the GPU viewer path in VecTask.render(): fetch the
        # completed simulation before stepping graphics, then render sensors.
        if env.device != 'cpu':
            gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        gym.render_all_camera_sensors(sim)
        img3 = gym.get_camera_image(sim, e0, cam3, gymapi.IMAGE_COLOR).reshape(cp.height, cp.width, 4)[..., :3].copy()
        imge = gym.get_camera_image(sim, e0, cam_ego, gymapi.IMAGE_COLOR).reshape(cp.height, cp.width, 4)[..., :3].copy()
        fmag = torch.linalg.norm(env.contact_forces[EI, :n_hand, :], dim=-1).cpu().numpy()
        fmax = max(fmax, float(fmag.max()))
        marks3, markse = [], []
        if os.environ.get('MARKERS', '1') == '1':
            R = quat_R(oq)
            for ml in MARK_LOCAL:
                wp = o + R @ ml
                marks3.append(project(cam3, wp)); markse.append(project(cam_ego, wp))
        frames.append((img3, imge, fmag, marks3, markse))
        tactile_log.append(fmag); qpos_log.append(env.rigid_body_states[EI, :n_hand, :3].cpu().numpy())
        action_log.append(mu[EI].cpu().numpy())
        obj_state = env.rigid_body_states[EI, obj_bi].cpu().numpy()   # [px,py,pz, qx,qy,qz,qw, vx..vz, wx,wy,wz]
        objrot_log.append(float(obj_state[12]))                      # angular velocity about z (rad/s)
        qx, qy, qz, qw = obj_state[3:7]                              # yaw about z from quaternion
        objquat_full_log.append(obj_state[3:7].copy())
        yaw = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
        objquat_log.append(float(yaw))
        objpos_log.append(obj_state[:3].copy()); handpos_log.append(env.rigid_body_states[EI, anchor, :3].cpu().numpy())
        dofpos_log.append(env.allegro_hand_dof_pos[EI].cpu().numpy().copy())   # 16 joint angles

    # Exact desired-axis rotation accumulated by the task reward/success implementation.
    cum_series = np.cumsum(np.array(signed_delta_log))
    # --- render video: 3rd | ego | tactile bar | rotation dial ---
    out = []
    for k, (img3, imge, fmag, marks3, markse) in enumerate(frames):
        fig, ax = plt.subplots(1, 4, figsize=(19, 5), gridspec_kw={'width_ratios': [1, 1, 1.1, 0.8]})
        ax[0].imshow(img3); ax[0].set_title('3rd-person'); ax[0].axis('off')
        ax[1].imshow(imge); ax[1].set_title('ego (palm-following cam)'); ax[1].axis('off')
        for mp, col in zip(marks3, MARK_COL):     # surface markers make the rotation visible
            if mp and 0 <= mp[0] < cp.width and 0 <= mp[1] < cp.height:
                ax[0].scatter([mp[0]], [mp[1]], s=140, c=col, edgecolors='k', zorder=5)
        for mp, col in zip(markse, MARK_COL):
            if mp and 0 <= mp[0] < cp.width and 0 <= mp[1] < cp.height:
                ax[1].scatter([mp[0]], [mp[1]], s=200, c=col, edgecolors='k', zorder=5)
        colors = cm.get_cmap('jet')(np.clip(fmag / fmax, 0, 1))
        ax[2].barh(range(n_hand), fmag, color=colors)
        ax[2].set_yticks(range(n_hand)); ax[2].set_yticklabels(labels, fontsize=6)
        ax[2].set_xlim(0, fmax); ax[2].invert_yaxis(); ax[2].set_title(f'per-body contact force (tactile)  step {k}')
        # rotation dial: arrow at current object yaw + cumulative degrees (proves the cylinder IS spinning)
        yc = float(cum_series[k]) if k < len(cum_series) else 0.0   # arrow tracks ACCUMULATED rotation
        cum = float(np.degrees(yc))
        th = np.linspace(0, 2*np.pi, 100)
        ax[3].plot(np.cos(th), np.sin(th), 'k', lw=1)
        ax[3].arrow(0, 0, 0.8*np.cos(yc), 0.8*np.sin(yc), head_width=0.12, head_length=0.12, fc='crimson', ec='crimson', lw=3)
        ax[3].set_xlim(-1.2, 1.2); ax[3].set_ylim(-1.2, 1.2); ax[3].set_aspect('equal'); ax[3].axis('off')
        ax[3].set_title(f'object yaw (about z)\ncumulative: {cum:+.0f}°')
        fig.tight_layout(); fig.canvas.draw()
        out.append(np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., :3].copy())
        plt.close(fig)
    vid = os.path.join(OUT, 'rollout.mp4')
    imageio.mimsave(vid, out, fps=20)
    # --- VERIFY rotation is real (not just translation) ---
    dt = float(env.control_freq_inv * env.dt)
    wz = np.array(objrot_log)                         # object angular velocity about z (rad/s)
    total_rot_deg = float(np.abs(wz).sum() * dt * 180 / np.pi)   # accumulated |rotation| about z
    objp = np.array(objpos_log); handp = np.array(handpos_log)
    obj_height_gap = float(np.mean(objp[:, 2] - handp[:, 2]))    # object above hand-base (z)
    hand_drift = float(np.linalg.norm(handp - handp[0], axis=1).mean())   # how much the HAND itself moved
    obj_xy_drift = float(np.linalg.norm(objp[:, :2] - objp[0, :2], axis=1).mean())
    print('=== ROTATION CHECK ===', flush=True)
    net_rot = float(np.degrees(cum_series[-1]))
    max_rot = float(np.degrees(np.max(cum_series)))
    success_idx = np.flatnonzero(cum_series >= float(env.success_rotation))
    success_step = int(success_idx[0] + 1) if len(success_idx) else -1
    reset_z = float(env.reset_z_threshold)
    min_object_z = float(objp[:, 2].min())
    done_idx = np.flatnonzero(np.array(done_log))
    first_done_step = int(done_idx[0] + 1) if len(done_idx) else -1
    early_done = bool(first_done_step > 0 and first_done_step < int(env.max_episode_length) - 1)
    stable_no_drop = bool(min_object_z >= reset_z and not early_done)
    verified_success = bool(success_step > 0 and stable_no_drop)
    print(f'  mean |ang_vel_z|: {np.abs(wz).mean():.3f} rad/s | EXACT desired-axis rotation: {net_rot:+.0f} deg (max {max_rot:+.0f}) over {N_STEPS} steps', flush=True)
    print(f'  success_step: {success_step} | min object z: {min_object_z:.4f} m | drop threshold: {reset_z:.4f} m | early_done: {early_done} | VERIFIED: {verified_success}', flush=True)
    print(f'  object-above-handbase gap: {obj_height_gap:.3f} m | hand-base drift: {hand_drift:.3f} m | object xy drift: {obj_xy_drift:.3f} m', flush=True)
    # JOINT motion: per-joint range of motion (deg) proves the fingers bend
    dofp = np.array(dofpos_log)                       # (T,16) joint angles (rad)
    rom_deg = np.degrees(dofp.max(0) - dofp.min(0))   # range of motion per joint
    print('  JOINT range-of-motion (deg) per DOF:', np.round(rom_deg, 0).astype(int).tolist(), flush=True)
    print(f'  -> mean {rom_deg.mean():.0f} deg, max {rom_deg.max():.0f} deg (fingers ARE flexing if >0)', flush=True)
    # joint-trajectory plot
    figj, axj = plt.subplots(figsize=(10, 5))
    for j in range(dofp.shape[1]):
        axj.plot(np.degrees(dofp[:, j]), label=f'dof{j}', lw=1)
    axj.set_xlabel('step'); axj.set_ylabel('joint angle (deg)'); axj.set_title('Shadow Hand joint angles over the rollout')
    axj.legend(ncol=4, fontsize=6); figj.tight_layout(); figj.savefig(os.path.join(OUT, 'joint_trajectories.png'), dpi=90); plt.close(figj)
    trajectory_path = os.path.join(OUT, 'trajectory_env0.npz')
    np.savez(trajectory_path, tactile=np.array(tactile_log), body_pos=np.array(qpos_log),
             action=np.array(action_log), body_names=np.array(body_names),
             obj_angvel_z=wz, obj_pos=objp, obj_quat=np.array(objquat_full_log),
             object_yaw=np.array(objquat_log), hand_pos=handp, joint_angles=dofp,
             physical_signed_delta_rad=np.array(signed_delta_log),
             task_signed_delta_rad=np.array(task_signed_delta_log),
             cumulative_rotation_rad=cum_series, done=np.array(done_log),
             full_rotation_success=(cum_series >= float(env.success_rotation)).astype(np.float32))
    summary = {
        'task': 'ShadowHandCubeRotation',
        'checkpoint': str(cfg.checkpoint),
        'seed': int(cfg.seed),
        'steps': int(N_STEPS),
        'success_rotation_rad': float(env.success_rotation),
        'rotation_metric': 'independent short-arc relative quaternion projected onto world -z',
        'exact_cumulative_rotation_rad': float(cum_series[-1]),
        'max_cumulative_rotation_rad': float(np.max(cum_series)),
        'success_step': success_step,
        'first_done_step': first_done_step,
        'min_object_z_m': min_object_z,
        'reset_z_threshold_m': reset_z,
        'early_done': early_done,
        'stable_no_drop': stable_no_drop,
        'verified_success': verified_success,
        'max_contact_force_n': float(fmax),
        'mean_joint_rom_deg': float(rom_deg.mean()),
        'video': vid,
        'trajectory': trajectory_path,
    }
    with open(os.path.join(OUT, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print('WROTE', vid, '| frames', len(out), '| max force', round(fmax, 2), '| n_hand', n_hand, flush=True)


if __name__ == '__main__':
    main()
