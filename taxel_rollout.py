"""Synced Hora rollout: runs the SAME policy/config/object as vis_s1.sh, but with an offscreen
camera so every rendered frame t is paired with the tactile at step t (perfect sync for the
combined viewer+taxel video). Nothing about the policy or env physics changes -- we only read
the contact forces + poses Hora already computes and render an offscreen camera.

Launch with the exact vis_s1.sh hydra overrides, e.g.:
  python taxel_rollout.py task=AllegroHandHora headless=True pipeline=gpu task.env.numEnvs=1 \
    test=True task.env.object.type=simple_tennis_ball train.algo=PPO \
    task.env.randomization.randomizeMass=False task.env.randomization.randomizeCOM=False \
    task.env.randomization.randomizeFriction=False task.env.randomization.randomizePDGains=False \
    task.env.randomization.randomizeScale=True train.ppo.priv_info=True \
    train.ppo.output_name=AllegroHandHora/hora_v0.0.2 \
    checkpoint=outputs/AllegroHandHora/hora_v0.0.2/stage1_nn/best.pth
"""
import isaacgym  # noqa: F401  (must precede torch)
import os, numpy as np, torch, hydra
from omegaconf import DictConfig, OmegaConf
from isaacgym import gymapi
from hora.tasks import isaacgym_task_map
from hora.utils.reformat import omegaconf_to_dict
from hora.algo.ppo.ppo import PPO

OmegaConf.register_new_resolver('eq', lambda x, y: x.lower() == y.lower())
OmegaConf.register_new_resolver('contains', lambda x, y: x.lower() in y.lower())
OmegaConf.register_new_resolver('if', lambda p, a, b: a if p else b)
OmegaConf.register_new_resolver('resolve_default', lambda d, a: d if a == '' else a)

OUT = os.environ.get('TAXEL_OUT', '/lp-dev/qianqian/taxel_run')
N = int(os.environ.get('TAXEL_STEPS', '260'))
WARM = int(os.environ.get('TAXEL_WARMUP', '40'))     # let the grasp settle before recording
W = int(os.environ.get('TAXEL_W', '620')); H = int(os.environ.get('TAXEL_H', '620'))


@hydra.main(config_name='config', config_path='configs')
def main(cfg: DictConfig):
    if cfg.checkpoint:
        from hydra.utils import to_absolute_path
        cfg.checkpoint = to_absolute_path(cfg.checkpoint)
    tc = omegaconf_to_dict(cfg.task)
    tc['enableCameraSensors'] = True
    env = isaacgym_task_map[cfg.task_name](tc, cfg.sim_device, cfg.graphics_device_id, headless=cfg.headless)
    gym, sim = env.gym, env.sim
    e0 = env.envs[0]
    agent = PPO(env, 'outputs/taxel_tmp', full_config=cfg)
    agent.restore_test(cfg.train.load_path); agent.set_eval()

    nb = env.num_allegro_hand_bodies                 # 17 hand bodies; object index = nb
    # camera: front, slightly elevated, framed on the object (matches the vis_s1 viewer feel)
    obs = env.reset()
    oc = env.rigid_body_states[0, nb, :3].cpu().numpy()   # object center (target the hand+ball cluster)
    ANGLES = {'top': ((0.002, -0.05, 0.34), (0.0, -0.05, 0.0)),
              'front': ((0.0, -0.34, 0.05), (0.0, 0.0, 0.0)),
              'side': ((0.34, 0.0, 0.05), (0.0, 0.0, 0.0)),
              'diag': ((0.22, 0.20, 0.15), (0.0, 0.0, 0.0))}
    _ang = os.environ.get('TAXEL_CAM_ANGLE')
    if os.environ.get('TAXEL_MULTICAM'):        # 4 cameras at once
        CVIEWS = [(nm, e, t) for nm, (e, t) in ANGLES.items()]
    elif _ang in ANGLES:                          # ONE named angle (render angles separately, fixed seed -> synced)
        CVIEWS = [(_ang, ANGLES[_ang][0], ANGLES[_ang][1])]
    elif os.environ.get('TAXEL_CAM', 'side') == 'topdown':
        CVIEWS = [('top', ANGLES['top'][0], ANGLES['top'][1])]
    else:
        CVIEWS = [('side', (0.15, 0.12, 0.05), (0.0, 0.0, -0.01))]
    cams = []
    for _nm, _e, _t in CVIEWS:
        cp = gymapi.CameraProperties(); cp.width = W; cp.height = H; cp.horizontal_fov = 50.0
        _cam = gym.create_camera_sensor(e0, cp)
        gym.set_camera_location(_cam, e0,
            gymapi.Vec3(float(oc[0]) + _e[0], float(oc[1]) + _e[1], float(oc[2]) + _e[2]),
            gymapi.Vec3(float(oc[0]) + _t[0], float(oc[1]) + _t[1], float(oc[2]) + _t[2]))
        cams.append(_cam)

    os.makedirs(OUT, exist_ok=True)
    os.makedirs(os.path.join(OUT, 'frames'), exist_ok=True)
    rbs_log, tac_log, jnt_log, wz_log = [], [], [], []
    dt = 1.0 / 60.0
    saved = 0
    for t in range(WARM + N):
        inp = {'obs': agent.running_mean_std(obs['obs']), 'priv_info': obs.get('priv_info', None)}
        mu = torch.clamp(agent.model.act_inference(inp), -1.0, 1.0)
        obs, r, done, info = env.step(mu)
        if t < WARM:
            continue
        # log tactile + poses for this exact step
        rbs_log.append(env.rigid_body_states[0, :nb + 1, :7].cpu().numpy().copy())
        tac_log.append(torch.linalg.norm(env.contact_forces[0, :nb + 1, :], dim=-1).cpu().numpy().copy())
        jnt_log.append(env.allegro_hand_dof_pos[0].cpu().numpy().copy())
        wz_log.append(float(env.rigid_body_states[0, nb, 12].cpu()))
        # render the same step from every camera
        gym.step_graphics(sim); gym.render_all_camera_sensors(sim)
        import imageio.v2 as imageio
        for _k, _cam in enumerate(cams):
            im = gym.get_camera_image(sim, e0, _cam, gymapi.IMAGE_COLOR).reshape(H, W, 4)[..., :3].copy()
            fn = ('c%d_%04d.png' % (_k, saved)) if len(cams) > 1 else ('f%04d.png' % saved)
            imageio.imwrite(os.path.join(OUT, 'frames', fn), im)
        saved += 1
        if saved % 40 == 0:
            print('rendered %d frames' % saved, flush=True)
    np.savez(os.path.join(OUT, 'tactile.npz'),
             rbs=np.array(rbs_log), tactile=np.array(tac_log),
             joints=np.array(jnt_log), objangvel_z=np.array(wz_log), num_hand_bodies=nb)
    print('DONE saved=%d frames + tactile.npz at %s' % (saved, OUT), flush=True)


if __name__ == '__main__':
    main()
