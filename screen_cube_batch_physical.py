"""One-episode batch audit using the corrected physical -z rotation delta."""

import isaacgym  # noqa: F401
import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from hora.algo.ppo.ppo import PPO
from hora.tasks import isaacgym_task_map
from hora.utils.reformat import omegaconf_to_dict


OmegaConf.register_new_resolver("eq", lambda x, y: x.lower() == y.lower())
OmegaConf.register_new_resolver("contains", lambda x, y: x.lower() in y.lower())
OmegaConf.register_new_resolver("if", lambda p, a, b: a if p else b)
OmegaConf.register_new_resolver("resolve_default", lambda d, a: d if a == "" else a)


@hydra.main(config_name="config", config_path="configs")
def main(cfg: DictConfig):
    if cfg.checkpoint:
        from hydra.utils import to_absolute_path
        cfg.checkpoint = to_absolute_path(cfg.checkpoint)
    task_cfg = omegaconf_to_dict(cfg.task)
    task_cfg["enableCameraSensors"] = True
    env = isaacgym_task_map[cfg.task_name](
        task_cfg, cfg.sim_device, cfg.graphics_device_id, cfg.headless
    )
    agent = PPO(env, "outputs/screen_batch_tmp", full_config=cfg)
    agent.restore_test(cfg.train.load_path)
    agent.set_eval()
    obs = env.reset()
    cumulative = torch.zeros(env.num_envs, device=env.device)
    max_cumulative = torch.zeros_like(cumulative)
    min_z = env.object_pos[:, 2].clone()
    dropped = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    # Stop before the timeout reset clears the per-episode task buffers.
    for _ in range(int(env.max_episode_length) - 1):
        inp = {
            "obs": agent.running_mean_std(obs["obs"]),
            "priv_info": obs.get("priv_info", None),
        }
        action = torch.clamp(agent.model.act_inference(inp), -1.0, 1.0)
        obs, _, done, _ = env.step(action)
        delta = env._signed_rotation_delta()
        cumulative += delta
        max_cumulative = torch.maximum(max_cumulative, cumulative)
        min_z = torch.minimum(min_z, env.object_pos[:, 2])
        # `done` includes the normal episode timeout. A drop is specifically
        # the task's object-height termination condition.
        dropped |= min_z < float(env.reset_z_threshold)
    success = (max_cumulative >= float(env.success_rotation)) & ~dropped
    order = torch.argsort(max_cumulative, descending=True)
    print(
        "PHYSICAL_BATCH_RESULT"
        f" envs={env.num_envs} successes={int(success.sum())}"
        f" success_rate={float(success.float().mean()):.6f}"
        f" max_rotation_rad={float(max_cumulative.max()):.6f}"
        f" median_rotation_rad={float(max_cumulative.median()):.6f}"
        f" drops={int(dropped.sum())}",
        flush=True,
    )
    for rank, idx in enumerate(order[:16].tolist()):
        print(
            "PHYSICAL_ENV"
            f" rank={rank} env={idx} max_rotation_rad={float(max_cumulative[idx]):.6f}"
            f" final_rotation_rad={float(cumulative[idx]):.6f}"
            f" min_z={float(min_z[idx]):.6f} dropped={bool(dropped[idx])}",
            flush=True,
        )


if __name__ == "__main__":
    main()
