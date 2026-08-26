# ShadowHandRWS — 5-finger Shadow in-hand rotation trained with the RWS / Touch Dexterity
# "teacher_cross" reward recipe (arxiv 2303.10880, YingYuan0414/in-hand-rotation), instead of Hora's.
#
# Design: reuse the WORKING Shadow sim (AllegroHandHora class + shadow asset + palm-up grasp cache +
# net-contact tactile). We ONLY override compute_reward to swap Hora's pose-diff-anchored reward for
# the RWS teacher_cross terms. Hora's own ShadowHandHora (v4) is left 100% untouched — this is a
# separate task/config, so both run side by side.
#
# RWS teacher_cross reward (non-ball path of compute_hand_reward_finger):
#   reward = spin_coef*spinned_theta + vel_coef*||obj_linvel|| + finger_coef*fingertip_proximity
#            + torque_coef*torque_penalty + work_coef*work_penalty  (+ fall_penalty on drop)
# Mapping onto Hora's already-computed quantities:
#   spinned_theta  -> rotate_reward   (axis-projected object angular velocity, clipped) * spin_coef
#   obj_linvel     -> (object_pos - object_pos_prev)/dt        (L2 norm, RWS uses L2)
#   fingertip prox -> clip(0.1/(4*dist+0.02),0,1).mean over the 5 Shadow tips  (RWS uses 4 Allegro tips)
#   torque_penalty -> (torques**2).sum        work_penalty -> ((torques*dof_vel_fd).sum)**2  (Hora's)
#   fall_penalty   -> added on Hora's check_termination (object dropped) reset
import torch
from isaacgym.torch_utils import quat_conjugate, quat_mul
from hora.tasks.allegro_hand_hora import AllegroHandHora, quat_to_axis_angle


class ShadowHandRWS(AllegroHandHora):
    def __init__(self, config, sim_device, graphics_device_id, headless):
        super().__init__(config, sim_device, graphics_device_id, headless)
        # Shadow 5 fingertips (ff/mf/rf/lf/th distal) and the object body (added right after the hand).
        self.ftip_body_idx = [7, 11, 15, 20, 25]
        self.obj_body_idx = self.num_allegro_hand_bodies
        r = self.config['env'].get('reward', {})
        self.rws_spin_coef = float(r.get('spinCoef', 1.0))
        self.rws_vel_coef = float(r.get('velCoef', -0.1))
        self.rws_torque_coef = float(r.get('torqueCoef', -3.0e-4))
        self.rws_work_coef = float(r.get('workCoef', -3.0e-4))
        self.rws_finger_coef = float(r.get('fingerCoef', 0.1))
        self.rws_fall_penalty = float(r.get('fallPenalty', -50.0))

    def compute_reward(self, actions):
        self.rot_axis_buf[:, -1] = -1  # spin about -z (same convention as Hora)

        # --- quantities Hora also computes (replicated so we don't depend on its reward internals) ---
        torque_penalty = (self.torques ** 2).sum(-1)
        work_penalty = ((self.torques * self.dof_vel_finite_diff).sum(-1)) ** 2
        angdiff = quat_to_axis_angle(quat_mul(self.object_rot, quat_conjugate(self.object_rot_prev)))
        object_angvel = angdiff / (self.control_freq_inv * self.dt)
        vec_dot = (object_angvel * self.rot_axis_buf).sum(-1)
        rotate_reward = torch.clip(vec_dot, max=self.angvel_clip_max, min=self.angvel_clip_min)
        object_linvel = ((self.object_pos - self.object_pos_prev) / (self.control_freq_inv * self.dt)).clone()

        # --- RWS fingertip-proximity reward (5 Shadow tips to the object) ---
        obj_pos = self.rigid_body_states[:, self.obj_body_idx, :3]              # (E, 3)
        ftip_pos = self.rigid_body_states[:, self.ftip_body_idx, :3]            # (E, 5, 3)
        dist = torch.sqrt(((obj_pos.unsqueeze(1) - ftip_pos) ** 2).sum(-1))     # (E, 5)
        distance_reward = torch.clip(0.1 / (4.0 * dist + 0.02), 0.0, 1.0).mean(-1) * self.rws_finger_coef

        # --- assemble the RWS teacher_cross reward ---
        spin_reward = self.rws_spin_coef * rotate_reward
        vel_reward = self.rws_vel_coef * torch.norm(object_linvel, dim=-1)
        reward = spin_reward + vel_reward + distance_reward \
            + torque_penalty * self.rws_torque_coef + work_penalty * self.rws_work_coef

        # fall: penalize + reset when the object drops (Hora's termination = object below height thresh)
        self.reset_buf[:] = self.check_termination(self.object_pos)
        reward = torch.where(self.reset_buf.bool(), reward + self.rws_fall_penalty, reward)
        self.rew_buf[:] = reward

        self.extras['spin_reward'] = spin_reward.mean()
        self.extras['vel_reward'] = vel_reward.mean()
        self.extras['finger_reward'] = distance_reward.mean()
        self.extras['torque_penalty'] = (torque_penalty * self.rws_torque_coef).mean()
        self.extras['work_penalty'] = (work_penalty * self.rws_work_coef).mean()
        self.extras['rotation_reward'] = rotate_reward.mean()
        self.extras['yaw'] = object_angvel[:, 2].mean()

        if self.evaluate:
            finished_episode_mask = self.reset_buf == 1
            self.stat_sum_rewards += self.rew_buf.sum()
            self.stat_sum_rotate_rewards += rotate_reward.sum()
            self.stat_sum_torques += self.torques.abs().sum()
            self.stat_sum_obj_linvel += (self.object_linvel ** 2).sum(-1).sum()
            self.stat_sum_episode_length += (self.reset_buf == 0).sum()
            self.env_evaluated += (self.reset_buf == 1).sum()
            self.env_timeout_counter[finished_episode_mask] += 1
