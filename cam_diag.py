"""Camera diagnostic: figure out why the hand isn't in the offscreen render.
Prints env origin + object global pos; renders 3 cameras (attached-to-base, close side, close top)."""
import isaacgym  # noqa
import os, numpy as np, torch, hydra
from omegaconf import DictConfig, OmegaConf
from isaacgym import gymapi
from hora.tasks import isaacgym_task_map
from hora.utils.reformat import omegaconf_to_dict
import imageio.v2 as imageio
OmegaConf.register_new_resolver('eq', lambda x, y: x.lower() == y.lower())
OmegaConf.register_new_resolver('contains', lambda x, y: x.lower() in y.lower())
OmegaConf.register_new_resolver('if', lambda p, a, b: a if p else b)
OmegaConf.register_new_resolver('resolve_default', lambda d, a: d if a == '' else a)
OUT = os.environ.get('CAM_OUT', '/lp-dev/qianqian/cam_diag')


@hydra.main(config_name='config', config_path='configs')
def main(cfg: DictConfig):
    tc = omegaconf_to_dict(cfg.task); tc['enableCameraSensors'] = True
    env = isaacgym_task_map[cfg.task_name](tc, cfg.sim_device, cfg.graphics_device_id, headless=cfg.headless)
    gym, sim = env.gym, env.sim
    e0 = env.envs[0]
    nb = env.num_allegro_hand_bodies
    # FORCE_CANON: set DOF to the grasp env's canonical (curled) pose + hold targets there, then settle.
    # (cam_diag's env.reset() does NOT run the grasp reset_idx, so without this the hand shows OPEN.)
    if os.environ.get('FORCE_CANON') and hasattr(env, 'canonical_pose'):
        from isaacgym import gymtorch
        cp = torch.tensor(env.canonical_pose, dtype=torch.float, device=env.device)
        env.allegro_hand_dof_pos[:] = cp
        env.allegro_hand_dof_vel[:] = 0
        env.dof_state.view(env.num_envs, -1, 2)[:, :env.num_allegro_hand_dofs, 0] = cp
        env.dof_state.view(env.num_envs, -1, 2)[:, :env.num_allegro_hand_dofs, 1] = 0
        env.gym.set_dof_state_tensor(env.sim, gymtorch.unwrap_tensor(env.dof_state))
        env.prev_targets[:, :env.num_allegro_hand_dofs] = cp
        env.cur_targets[:, :env.num_allegro_hand_dofs] = cp
        print('FORCED canonical pose', flush=True)
    _settle = int(os.environ.get('DIAG_SETTLE', '30'))   # 0 = render the pure reset (cached-grasp) pose
    for _ in range(_settle):
        env.step(torch.zeros((env.num_envs, env.num_actions), device=env.device))
    org = gym.get_env_origin(e0)
    oc = env.rigid_body_states[0, nb, :3].cpu().numpy()
    bl = env.rigid_body_states[0, 0, :3].cpu().numpy()
    print('ENV ORIGIN:', org.x, org.y, org.z, flush=True)
    print('OBJ GLOBAL:', np.round(oc, 3), '| BASE GLOBAL:', np.round(bl, 3), flush=True)
    rb = env.rigid_body_states[0, :, :3].cpu().numpy()
    palm = rb[3]
    for nm, idx in [('ff', 7), ('mf', 11), ('rf', 15), ('lf', 20), ('th', 25)]:
        print('TIP %s idx%d:' % (nm, idx), np.round(rb[idx], 3), '| rel-palm:', np.round(rb[idx] - palm, 3), flush=True)
    print('PALM idx3:', np.round(palm, 3), '| rel-base:', np.round(palm - bl, 3), flush=True)
    ftips = np.array([rb[i] for i in [7, 11, 15, 20, 25]])
    faxis = ftips.mean(0) - palm
    print('FINGER_AXIS (mean tip - palm):', np.round(faxis, 4), '| dir:', np.round(faxis / (np.linalg.norm(faxis) + 1e-9), 3), flush=True)
    spread = rb[20] - rb[7]          # lf_tip - ff_tip (across the fingers)
    pn = np.cross(faxis, spread); pn = pn / (np.linalg.norm(pn) + 1e-9)
    if pn[2] < 0:
        pn = -pn                     # orient toward the palm face (up-ish)
    print('PALM_NORMAL dir:', np.round(pn, 3), '(want ~[0,0,1] = palm facing UP)', flush=True)
    os.makedirs(OUT, exist_ok=True)
    W = H = 600

    def shoot(name, eye, tgt, attach_body=None):
        cp = gymapi.CameraProperties(); cp.width = W; cp.height = H; cp.horizontal_fov = 60
        cam = gym.create_camera_sensor(e0, cp)
        if attach_body is not None:
            bh = gym.get_actor_rigid_body_handle(e0, gym.get_actor_handle(e0, 0), attach_body)
            t = gymapi.Transform(); t.p = gymapi.Vec3(*eye)
            gym.attach_camera_to_body(cam, e0, bh, t, gymapi.FOLLOW_TRANSFORM)
        else:
            gym.set_camera_location(cam, e0, gymapi.Vec3(*eye), gymapi.Vec3(*tgt))
        gym.step_graphics(sim); gym.render_all_camera_sensors(sim)
        im = gym.get_camera_image(sim, e0, cam, gymapi.IMAGE_COLOR).reshape(H, W, 4)[..., :3].copy()
        imageio.imwrite(os.path.join(OUT, name + '.png'), im)
        print('shot', name, 'nonzero-nonfloor px frac=%.3f' % ((im.sum(-1) < 120).mean()), flush=True)

    o = oc
    mid = (float(bl[0]), 0.5 * (float(bl[1]) + float(o[1])), 0.5 * (float(bl[2]) + float(o[2])))  # hand center
    # 4-view montage set (all framed on the whole hand so the ball's height vs palm is clear)
    shoot('topdown', (mid[0] + 0.001, mid[1] + 0.001, mid[2] + 0.42), (mid[0], mid[1], mid[2] - 0.1))
    shoot('side', (mid[0] + 0.42, mid[1], mid[2] + 0.04), mid)          # profile from +X (shows ball-vs-palm height)
    shoot('front', (mid[0], float(o[1]) - 0.40, float(o[2]) + 0.04), (mid[0], mid[1], float(o[2])))  # fingertip end
    shoot('angled', (mid[0] + 0.30, mid[1] + 0.28, mid[2] + 0.18), mid)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
