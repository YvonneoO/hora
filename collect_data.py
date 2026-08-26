# Data-collection rollout for the pretrained Hora policy: 3rd-person + ego RGB video + per-hand-body
# contact force (tactile, "16-FSR"-style) + proprioception. Place at hora/ root (hydra config_path='configs').
import isaacgym  # noqa (must be before torch)
import os, numpy as np, torch, hydra
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
    for i, nm in enumerate(body_names):
        if any(k in nm.lower() for k in ['palm', 'base', 'link_0', 'wrist']):
            anchor = i; break
    anchor_handle = gym.get_actor_rigid_body_handle(e0, hand, anchor)
    print('ego anchor body:', body_names[anchor], flush=True)

    # --- cameras --- object body index in rigid_body_states = right after the hand bodies
    obj_bi = n_hand
    cp = gymapi.CameraProperties(); cp.width = 480; cp.height = 480; cp.enable_tensors = False
    hand_state = env.rigid_body_states[EI, anchor, :3].cpu().numpy()
    cx, cy, cz = float(hand_state[0]), float(hand_state[1]), float(hand_state[2])
    cam3 = gym.create_camera_sensor(e0, cp)     # HORA's EXACT default viewer camera (env line 53-54)
    _e = [float(v) for v in os.environ.get('CAM3_EYE', '0.0,0.4,1.5').split(',')]
    _t = [float(v) for v in os.environ.get('CAM3_TGT', '0.0,0.0,0.5').split(',')]
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
    dofpos_log = []   # the 16 Allegro JOINT ANGLES each step (to prove the fingers actually bend)
    fmax = 1e-6
    for t in range(N_STEPS):
        inp = {'obs': agent.running_mean_std(obs['obs']), 'priv_info': obs.get('priv_info', None)}
        mu = torch.clamp(agent.model.act_inference(inp), -1.0, 1.0)
        obs, r, done, info = env.step(mu)
        # ego close-up: track the object each step (eye just in front/above the object, looking at it)
        o, oq = obj_pose()
        gym.set_camera_location(cam_ego, e0, gymapi.Vec3(float(o[0]) + 0.11, float(o[1]) + 0.07, float(o[2]) + 0.07),
                                gymapi.Vec3(float(o[0]), float(o[1]), float(o[2])))
        gym.render_all_camera_sensors(sim)
        img3 = gym.get_camera_image(sim, e0, cam3, gymapi.IMAGE_COLOR).reshape(cp.height, cp.width, 4)[..., :3].copy()
        imge = gym.get_camera_image(sim, e0, cam_ego, gymapi.IMAGE_COLOR).reshape(cp.height, cp.width, 4)[..., :3].copy()
        fmag = torch.linalg.norm(env.contact_forces[EI, :n_hand, :], dim=-1).cpu().numpy()
        fmax = max(fmax, float(fmag.max()))
        marks3, markse = [], []
        if os.environ.get('MARKERS', '0') == '1':         # markers off by default (use a textured object instead)
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
        yaw = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
        objquat_log.append(float(yaw))
        objpos_log.append(obj_state[:3].copy()); handpos_log.append(env.rigid_body_states[EI, anchor, :3].cpu().numpy())
        dofpos_log.append(env.allegro_hand_dof_pos[EI].cpu().numpy().copy())   # 16 joint angles

    # cumulative rotation by INTEGRATING angular velocity about z (robust; matches the per-env diagnostic).
    cum_series = np.cumsum(np.array(objrot_log)) * (1.0 / 60.0)   # radians
    # --- render video: 3rd | ego | tactile bar | rotation dial ---
    out = []
    for k, (img3, imge, fmag, marks3, markse) in enumerate(frames):
        fig, ax = plt.subplots(1, 4, figsize=(19, 5), gridspec_kw={'width_ratios': [1, 1, 1.1, 0.8]})
        ax[0].imshow(img3); ax[0].set_title('3rd-person'); ax[0].axis('off')
        ax[1].imshow(imge); ax[1].set_title('ego (palm cam)'); ax[1].axis('off')
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
    vid = os.path.join(OUT, 'hora_rollout.mp4')
    imageio.mimsave(vid, out, fps=20)
    # --- VERIFY rotation is real (not just translation) ---
    dt = float(getattr(env, 'dt', getattr(env, 'control_dt', 1/20.0)))
    wz = np.array(objrot_log)                         # object angular velocity about z (rad/s)
    total_rot_deg = float(np.abs(wz).sum() * dt * 180 / np.pi)   # accumulated |rotation| about z
    objp = np.array(objpos_log); handp = np.array(handpos_log)
    obj_height_gap = float(np.mean(objp[:, 2] - handp[:, 2]))    # object above hand-base (z)
    hand_drift = float(np.linalg.norm(handp - handp[0], axis=1).mean())   # how much the HAND itself moved
    obj_xy_drift = float(np.linalg.norm(objp[:, :2] - objp[0, :2], axis=1).mean())
    print('=== ROTATION CHECK ===', flush=True)
    net_rot = float(np.degrees(np.sum(wz) / 60.0))   # integrate ang-vel-z (robust, matches diagnostic)
    print(f'  mean |ang_vel_z|: {np.abs(wz).mean():.3f} rad/s | NET rotation (integrated): {net_rot:+.0f} deg over {N_STEPS} steps', flush=True)
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
    axj.set_xlabel('step'); axj.set_ylabel('joint angle (deg)'); axj.set_title('Allegro 16 joint angles over the rollout')
    axj.legend(ncol=4, fontsize=6); figj.tight_layout(); figj.savefig(os.path.join(OUT, 'joint_trajectories.png'), dpi=90); plt.close(figj)
    np.savez(os.path.join(OUT, 'hora_data.npz'), tactile=np.array(tactile_log), body_pos=np.array(qpos_log),
             action=np.array(action_log), body_names=np.array(body_names),
             obj_angvel_z=wz, obj_pos=objp, hand_pos=handp, joint_angles=dofp)
    print('WROTE', vid, '| frames', len(out), '| max force', round(fmax, 2), '| n_hand', n_hand, flush=True)


if __name__ == '__main__':
    main()
