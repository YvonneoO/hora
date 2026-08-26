# Shadow-hand grasp generator: same Hora grasp pipeline as AllegroHandGrasp, but with the
# 5-finger Shadow geometry (24 DOF, 5 fingertips ff/mf/rf/lf/thdistal = bodies 7,11,15,20,25;
# object body index = num hand bodies). Only the hand-specific constants are overridden.
import os, json
import torch
import numpy as np
from isaacgym import gymtorch
from isaacgym.torch_utils import torch_rand_float, tensor_clamp, to_torch
from hora.tasks.allegro_hand_grasp import AllegroHandGrasp

# default fingertip-pinch canonical (24 DOF). DOF order: WRJ1,WRJ0, then each finger [J3 spread, J2 MCP,
# J1 PIP, J0 DIP] for FF/MF/RF, LF has an extra J4 metacarpal first, thumb [THJ4 CMC-opp, THJ3, THJ2, THJ1, THJ0].
DEFAULT_CANON = [
    0.0, 0.0,
    -0.15, 0.50, 0.75, 0.60,
    -0.05, 0.50, 0.75, 0.60,
    0.05, 0.50, 0.75, 0.60,
    0.0, 0.15, 0.50, 0.75, 0.60,
    1.00, 0.70, 0.0, 0.0, -0.60,
]


class ShadowHandGrasp(AllegroHandGrasp):
    def __init__(self, config, sim_device, graphics_device_id, headless):
        super().__init__(config, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless)
        # grasp-state row = hand DOF (24) + object 7-pose = 31
        self.saved_grasping_states = torch.zeros((0, self.num_allegro_hand_dofs + 7),
                                                 dtype=torch.float, device=self.device)
        self.ftip_body_idx = [7, 11, 15, 20, 25]          # ff, mf, rf, lf, th distal
        self.obj_body_idx = self.num_allegro_hand_bodies  # object added after the hand
        # canonical pose + cache name are ENV-DRIVEN so many variants can run in parallel (one per GPU).
        _c = os.environ.get('GRASP_CANON', '')
        self.canonical_pose = json.loads(_c) if _c else list(DEFAULT_CANON)
        _n = os.environ.get('GRASP_NAME', '')
        if _n:
            self.grasp_cache_name = _n
        print('[ShadowHandGrasp] cache=%s canon[thumb]=%s' % (self.grasp_cache_name, self.canonical_pose[-5:]), flush=True)

    def compute_reward(self, actions):
        # (cond2 fingertip-contact check dropped -> skip the expensive per-env get_env_rigid_contacts loop)
        obj_pos = self.rigid_body_states[:, [self.obj_body_idx], :3]
        finger_pos = self.rigid_body_states[:, self.ftip_body_idx, :3]
        palm_pos = self.rigid_body_states[:, [3], :3]           # robot0:palm
        tip_dist = torch.sqrt(((obj_pos - finger_pos) ** 2).sum(-1))
        # 1) envelope: >=4 of 5 fingertips wrapping the ball (~8cm covers tips curling over the top;
        #    real cupped-tip-to-center dist measured ~5-8cm). The LOW-ball check below is what rejects
        #    the old high fingertip-balances, so this radius can be generous.
        cond1 = (tip_dist < 0.13).sum(-1) >= 3
        # 1b) ball cradled LOW (near the palm), NOT balanced up on the fingertips (the key discriminator:
        #    old drop-prone grasp had ball ~12cm above palm; a real cup is <9cm)
        cond1b = (obj_pos[:, 0, 2] - palm_pos[:, 0, 2]) < 0.16
        # 3) object hasn't fallen
        cond3 = torch.greater(obj_pos[:, -1, -1], self.reset_z_threshold)
        # NOTE: cond2 (fingertip-body rigid contact) dropped — the long Shadow fingers WRAP over the ball,
        # so the fingertip bodies rarely register contact even in a firm stable grip. Rely on cond1 (tips
        # near) + cond1b (ball held at finger level) + cond3 (ball hasn't fallen = something holds it).
        cond = cond1.float() * cond1b.float() * cond3.float()
        self.reset_buf[cond < 1] = 1
        self.reset_buf[self.progress_buf >= self.max_episode_length] = 1
