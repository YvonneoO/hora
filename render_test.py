# Rendering sanity test: force the hand to OPEN vs CURLED poses and render each. If the fingers look different,
# rendering reflects joint state correctly (so any "static" look in the rollout = small policy motion, not a bug).
# Also colors the object so it's not white.
import isaacgym
import os, numpy as np, torch, hydra
from omegaconf import DictConfig, OmegaConf
from isaacgym import gymapi
from hora.tasks import isaacgym_task_map
from hora.utils.reformat import omegaconf_to_dict
from hora.algo.ppo.ppo import PPO
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import imageio.v2 as imageio
OmegaConf.register_new_resolver('eq', lambda x, y: x.lower() == y.lower())
OmegaConf.register_new_resolver('contains', lambda x, y: x.lower() in y.lower())
OmegaConf.register_new_resolver('if', lambda p, a, b: a if p else b)
OmegaConf.register_new_resolver('resolve_default', lambda d, a: d if a == '' else a)
OUT = '/lp-dev/qianqian/hora_render_test'


@hydra.main(config_name='config', config_path='configs')
def main(cfg: DictConfig):
    if cfg.checkpoint:
        from hydra.utils import to_absolute_path
        cfg.checkpoint = to_absolute_path(cfg.checkpoint)
    tc = omegaconf_to_dict(cfg.task); tc['enableCameraSensors'] = True
    env = isaacgym_task_map[cfg.task_name](tc, cfg.sim_device, cfg.graphics_device_id, cfg.headless)
    gym, sim = env.gym, env.sim
    e0 = env.envs[0]
    hand = gym.find_actor_handle(e0, 'hand'); obj = gym.find_actor_handle(e0, 'object')
    # COLOR the object bright orange, and each finger link a distinct color (so bending is obvious)
    gym.set_rigid_body_color(e0, obj, 0, gymapi.MESH_VISUAL, gymapi.Vec3(1.0, 0.5, 0.0))
    fcolors = [gymapi.Vec3(*c) for c in [(0.9,0.1,0.1),(0.1,0.8,0.1),(0.1,0.3,1.0),(0.9,0.8,0.1)]]
    nb = gym.get_actor_rigid_body_count(e0, hand)
    for bi in range(nb):
        gym.set_rigid_body_color(e0, hand, bi, gymapi.MESH_VISUAL, fcolors[(bi-1)//4 % 4] if bi > 0 else gymapi.Vec3(0.4,0.4,0.4))
    env.reset()
    cx, cy, cz = [float(v) for v in env.rigid_body_states[0, 0, :3].cpu().numpy()]
    cp = gymapi.CameraProperties(); cp.width = 520; cp.height = 520
    cam = gym.create_camera_sensor(e0, cp)
    gym.set_camera_location(cam, e0, gymapi.Vec3(cx + 0.16, cy + 0.14, cz + 0.11), gymapi.Vec3(cx, cy, cz + 0.02))
    nd = env.num_allegro_hand_dofs
    lo = env.allegro_hand_dof_lower_limits; hi = env.allegro_hand_dof_upper_limits
    poses = {'OPEN (all 0)': torch.zeros(nd, device=env.device),
             'MID (0.4*upper)': 0.4 * hi[:nd],
             'CURLED (0.85*upper)': 0.85 * hi[:nd]}
    imgs = []
    for name, p in poses.items():
        env.allegro_hand_dof_pos[:, :] = p[None]
        env.dof_state.view(env.num_envs, -1, 2)[:, :nd, 0] = p[None]
        gym.set_dof_state_tensor(sim, gymtorch_unwrap(env.dof_state))
        gym.simulate(sim); gym.fetch_results(sim, True); gym.step_graphics(sim)
        gym.render_all_camera_sensors(sim)
        im = gym.get_camera_image(sim, e0, cam, gymapi.IMAGE_COLOR).reshape(cp.height, cp.width, 4)[..., :3].copy()
        imgs.append((name, im))
        print('rendered', name, flush=True)
    os.makedirs(OUT, exist_ok=True)
    fig, ax = plt.subplots(1, len(imgs), figsize=(6 * len(imgs), 6))
    for a, (name, im) in zip(ax, imgs):
        a.imshow(im); a.set_title(name); a.axis('off')
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'render_test.png'), dpi=90); plt.close(fig)
    print('WROTE', os.path.join(OUT, 'render_test.png'), flush=True)


def gymtorch_unwrap(t):
    from isaacgym import gymtorch
    return gymtorch.unwrap_tensor(t)


if __name__ == '__main__':
    main()
