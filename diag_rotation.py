# Diagnostic: does the pretrained policy actually ROTATE, per-env? (no cameras, fast). Reports each env's
# net rotation (integrating object ang-vel about z) so we know if env 0 was just a bad pick or the policy is off.
import isaacgym
import os, numpy as np, torch, hydra
from omegaconf import DictConfig, OmegaConf
from hora.tasks import isaacgym_task_map
from hora.utils.reformat import omegaconf_to_dict
from hora.algo.ppo.ppo import PPO
OmegaConf.register_new_resolver('eq', lambda x, y: x.lower() == y.lower())
OmegaConf.register_new_resolver('contains', lambda x, y: x.lower() in y.lower())
OmegaConf.register_new_resolver('if', lambda p, a, b: a if p else b)
OmegaConf.register_new_resolver('resolve_default', lambda d, a: d if a == '' else a)
N = int(os.environ.get('DIAG_STEPS', '300'))


@hydra.main(config_name='config', config_path='configs')
def main(cfg: DictConfig):
    if cfg.checkpoint:
        from hydra.utils import to_absolute_path
        cfg.checkpoint = to_absolute_path(cfg.checkpoint)
    env = isaacgym_task_map[cfg.task_name](omegaconf_to_dict(cfg.task), cfg.sim_device, cfg.graphics_device_id, cfg.headless)
    agent = PPO(env, 'outputs/diag_tmp', full_config=cfg); agent.restore_test(cfg.train.load_path); agent.set_eval()
    ne = env.num_envs
    obj_bi = env.num_allegro_hand_bodies
    dt = 1.0 / 60.0
    obs = env.reset()
    wz_sum = torch.zeros(ne, device=env.device)     # net signed rotation (rad, integrated)
    wz_abs = torch.zeros(ne, device=env.device)      # accumulated |rotation|
    for t in range(N):
        inp = {'obs': agent.running_mean_std(obs['obs']), 'priv_info': obs.get('priv_info', None)}
        mu = torch.clamp(agent.model.act_inference(inp), -1.0, 1.0)
        obs, r, done, info = env.step(mu)
        wz = env.rigid_body_states[:, obj_bi, 12]    # ang vel about z, all envs
        wz_sum += wz * dt; wz_abs += wz.abs() * dt
    net = np.degrees(wz_sum.cpu().numpy())
    acc = np.degrees(wz_abs.cpu().numpy())
    print('=== PER-ENV ROTATION over %d steps (deg) ===' % N, flush=True)
    print('  net rotation per env:', np.round(net, 0).astype(int).tolist(), flush=True)
    print('  |accumulated| per env:', np.round(acc, 0).astype(int).tolist(), flush=True)
    print(f'  mean |net|: {np.abs(net).mean():.0f} deg | mean accumulated: {acc.mean():.0f} deg | envs with |net|>180: {(np.abs(net) > 180).sum()}/{ne}', flush=True)
    print(f'  BEST env (max accumulated): env {int(acc.argmax())} = {acc.max():.0f} deg', flush=True)


if __name__ == '__main__':
    main()
