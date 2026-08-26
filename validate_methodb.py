"""Method-B validation: run the v4 policy (CPU pipeline) and log REAL per-contact points via
gym.get_env_rigid_contacts (position on the hand body + normal force 'lambda'), alongside the per-body
NET force we use now. Saves methodb.npz for offline comparison. No asset/policy change."""
import isaacgym  # noqa
import os, numpy as np, torch, hydra
from omegaconf import DictConfig, OmegaConf
from hora.tasks import isaacgym_task_map
from hora.utils.reformat import omegaconf_to_dict
from hora.algo.ppo.ppo import PPO
OmegaConf.register_new_resolver('eq', lambda x, y: x.lower() == y.lower())
OmegaConf.register_new_resolver('contains', lambda x, y: x.lower() in y.lower())
OmegaConf.register_new_resolver('if', lambda p, a, b: a if p else b)
OmegaConf.register_new_resolver('resolve_default', lambda d, a: d if a == '' else a)
OUT = os.environ.get('MB_OUT', '/lp-dev/qianqian/methodb')
N = int(os.environ.get('MB_STEPS', '40')); WARM = int(os.environ.get('MB_WARM', '15'))


def qapply(q, v):
    xyz = q[:3]; w = q[3]; t = 2.0 * np.cross(xyz, v)
    return v + w * t + np.cross(xyz, t)


def vec3(x):
    try:
        return np.array([float(x['x']), float(x['y']), float(x['z'])])
    except Exception:
        return np.array([float(x[0]), float(x[1]), float(x[2])])


@hydra.main(config_name='config', config_path='configs')
def main(cfg: DictConfig):
    if cfg.checkpoint:
        from hydra.utils import to_absolute_path
        cfg.checkpoint = to_absolute_path(cfg.checkpoint)
    env = isaacgym_task_map[cfg.task_name](omegaconf_to_dict(cfg.task), cfg.sim_device, cfg.graphics_device_id, headless=True)
    agent = PPO(env, 'outputs/mb_tmp', full_config=cfg); agent.restore_test(cfg.train.load_path); agent.set_eval()
    assert env.device == 'cpu', 'rigid_contacts needs pipeline=cpu'
    nb = env.num_allegro_hand_bodies; obj = nb
    obs = env.reset()
    net_log, rbs_log, cont_log = [], [], []
    printed = False
    for t in range(WARM + N):
        inp = {'obs': agent.running_mean_std(obs['obs']), 'priv_info': obs.get('priv_info', None)}
        mu = torch.clamp(agent.model.act_inference(inp), -1.0, 1.0)
        obs, r, done, info = env.step(mu)
        if t < WARM:
            continue
        net = torch.linalg.norm(env.contact_forces[0, :nb + 1, :], dim=-1).cpu().numpy().copy()
        rbs = env.rigid_body_states[0, :nb + 1, :7].cpu().numpy().copy()
        contacts = env.gym.get_env_rigid_contacts(env.envs[0])
        if not printed:
            print('CONTACT FIELDS:', contacts.dtype.names, flush=True); printed = True
        rows = []
        for c in contacts:
            b0 = int(c['body0']); b1 = int(c['body1']); lam = float(c['lambda'])
            if b0 == obj and 0 <= b1 < nb:
                hb, lp = b1, vec3(c['localPos1'])
            elif b1 == obj and 0 <= b0 < nb:
                hb, lp = b0, vec3(c['localPos0'])
            else:
                continue
            world = rbs[hb, :3] + qapply(rbs[hb, 3:7], lp)
            rows.append([hb, world[0], world[1], world[2], lam])
        net_log.append(net); rbs_log.append(rbs)
        cont_log.append(np.array(rows) if rows else np.zeros((0, 5)))
    os.makedirs(OUT, exist_ok=True)
    np.savez(os.path.join(OUT, 'methodb.npz'), net=np.array(net_log), rbs=np.array(rbs_log),
             contacts=np.array(cont_log, dtype=object), obj=obj, nb=nb)
    # quick console summary: per-body net vs sum-of-contact-forces
    allc = np.vstack([c for c in cont_log if len(c)])
    print('DONE steps=%d total_contacts=%d' % (len(net_log), allc.shape[0]), flush=True)
    for b in sorted(set(allc[:, 0].astype(int))):
        m = allc[:, 0].astype(int) == b
        print('  body %2d: n_contacts=%4d  sum|lambda|=%.2f  mean_net=%.2f' %
              (b, m.sum(), allc[m, 4].sum(), np.array(net_log)[:, b].sum()), flush=True)


if __name__ == '__main__':
    main()
