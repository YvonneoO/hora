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
    # Match the renderer's environment construction so seeded grasp sampling is identical.
    task_cfg["enableCameraSensors"] = True
    env = isaacgym_task_map[cfg.task_name](
        task_cfg, cfg.sim_device, cfg.graphics_device_id, cfg.headless
    )
    agent = PPO(env, "outputs/screen_tmp", full_config=cfg)
    agent.restore_test(cfg.train.load_path)
    agent.set_eval()

    obs = env.reset()
    exact_cumulative = 0.0
    max_cumulative = 0.0
    min_object_z = float("inf")
    success_step = -1
    first_done_step = -1
    for step in range(int(env.max_episode_length)):
        inp = {
            "obs": agent.running_mean_std(obs["obs"]),
            "priv_info": obs.get("priv_info", None),
        }
        action = torch.clamp(agent.model.act_inference(inp), -1.0, 1.0)
        obs, _, done, info = env.step(action)
        signed_delta = float(info["signed_delta_rad"])
        exact_cumulative += signed_delta
        max_cumulative = max(max_cumulative, exact_cumulative)
        min_object_z = min(min_object_z, float(env.object_pos[0, 2]))
        if success_step < 0 and float(info["full_rotation_success"]) >= 0.5:
            success_step = step + 1
        if bool(done[0]) and first_done_step < 0:
            first_done_step = step + 1
            break

    print(
        "SCREEN_RESULT"
        f" exact_cumulative_rad={exact_cumulative:.6f}"
        f" max_cumulative_rad={max_cumulative:.6f}"
        f" success_step={success_step}"
        f" first_done_step={first_done_step}"
        f" min_object_z={min_object_z:.6f}"
        f" reset_z_threshold={float(env.reset_z_threshold):.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
