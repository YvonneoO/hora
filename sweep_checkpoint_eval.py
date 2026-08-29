#!/usr/bin/env python3
"""Re-evaluate every periodic checkpoint from a completed run to reconstruct a
success-rate-vs-training-step curve, using the correctly-masked --evaluate
stat block (unaffected by the reset_buf-as-index bug fixed in
allegro_hand_hora.py). Each checkpoint gets its own short train.py subprocess
(test=True, task.on_evaluation=True, small evalTargetEpisodes) since the
network's obs-dim/priv_info branch must match how that checkpoint was
trained.

Usage: python sweep_checkpoint_eval.py <ckpt_dir> <out_csv> [--priv_info True|False]
    [--coarse_tactile True|False] [--target_episodes N] [--num_envs N]
"""
import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

CKPT_RE = re.compile(r"ep_(\d+)_step_(\d+)M_reward_([\d.]+)\.pth")
SUCCESS_RE = re.compile(r"success rate:\s*([\d.]+)%")
REWARD_RE = re.compile(r"reward:\s*([\d.-]+)\s*\|")
EPLEN_RE = re.compile(r"eps length:\s*([\d.]+)\s*\|")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_dir")
    ap.add_argument("out_csv")
    ap.add_argument("--priv_info", default="False")
    ap.add_argument("--coarse_tactile", default="False")
    ap.add_argument("--target_episodes", type=int, default=200)
    ap.add_argument("--num_envs", type=int, default=16384)
    args = ap.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    hora_root = Path(__file__).resolve().parent
    python_bin = sys.executable

    ckpts = []
    for f in sorted(ckpt_dir.glob("ep_*.pth")):
        m = CKPT_RE.match(f.name)
        if m:
            ep, step_m, reward_at_save = m.groups()
            ckpts.append((int(ep), int(step_m) * 1_000_000, float(reward_at_save), f))
    ckpts.sort(key=lambda t: t[0])

    print(f"Found {len(ckpts)} periodic checkpoints in {ckpt_dir}")
    rows = []
    for ep, step, reward_at_save, ckpt_path in ckpts:
        cmd = [
            python_bin, str(hora_root / "train.py"),
            "task=ShadowHandHora", "headless=True", "pipeline=gpu", "test=True",
            "task.on_evaluation=True",
            f"task.env.evalTargetEpisodes={args.target_episodes}",
            f"task.env.numEnvs={args.num_envs}",
            "task.env.object.type=simple_tennis_ball",
            "task.env.grasp_cache_name=v4_spread",
            "task.env.randomization.randomizeMass=False",
            "task.env.randomization.randomizeCOM=False",
            "task.env.randomization.randomizeFriction=False",
            "task.env.randomization.randomizePDGains=False",
            "task.env.randomization.randomizeScale=True",
            "task.env.randomization.randomizeScaleList=[0.7]",
            "task.env.forceScale=0", "task.env.randomForceProbScalar=0",
            f"task.env.coarseTactile={args.coarse_tactile}",
            "train.algo=PPO", f"train.ppo.priv_info={args.priv_info}",
            f"checkpoint={ckpt_path}",
            "seed=0", "sim_device=cuda:0", "rl_device=cuda:0", "graphics_device_id=0",
        ]
        print(f"--- ep {ep} (step {step}) ---", flush=True)
        proc = subprocess.run(cmd, cwd=str(hora_root), capture_output=True, text=True, timeout=600)
        out = proc.stdout + proc.stderr
        success_matches = SUCCESS_RE.findall(out)
        reward_matches = REWARD_RE.findall(out)
        eplen_matches = EPLEN_RE.findall(out)
        if not success_matches:
            print(f"  NO SUCCESS LINE FOUND (returncode={proc.returncode}); tail:")
            print("  " + "\n  ".join(out.splitlines()[-15:]))
            success_rate = None
        else:
            success_rate = float(success_matches[-1])
        final_reward = float(reward_matches[-1]) if reward_matches else None
        final_eplen = float(eplen_matches[-1]) if eplen_matches else None
        print(f"  success_rate={success_rate} reward={final_reward} eplen={final_eplen}", flush=True)
        rows.append({
            "epoch": ep, "agent_steps": step, "reward_at_save": reward_at_save,
            "eval_success_rate_pct": success_rate, "eval_reward": final_reward,
            "eval_episode_length": final_eplen,
        })
        with open(args.out_csv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print(f"DONE. Wrote {len(rows)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()
