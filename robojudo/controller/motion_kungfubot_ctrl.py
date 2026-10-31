# from .motion_lib import MotionLibReal
import logging
import time

import torch

from robojudo.controller import ctrl_registry
from robojudo.controller.ctrl_cfgs import MotionKungfuBotCtrlCfg
from robojudo.controller.motion_ctrl import MotionCtrl
from robojudo.utils.util_func_torch import (
    get_euler_xyz_in_tensor,
    quat_apply,
    quat_inverse,
    quat_rotate_inverse,
)

logger = logging.getLogger(__name__)


@ctrl_registry.register
class MotionKungfuBotCtrl(MotionCtrl):
    cfg_ctrl: MotionKungfuBotCtrlCfg

    def __init__(self, cfg_ctrl, env, device="cpu"):
        self._pack_motions(cfg_ctrl)
        super().__init__(cfg_ctrl=cfg_ctrl, env=env, device=device)

        self.future_max_steps = self.cfg_ctrl.future_max_steps
        self.future_num_steps = self.cfg_ctrl.future_num_steps

        self.anchor_index = self.cfg_ctrl.anchor_index
        self.key_body_id = self.cfg_ctrl.key_body_id

        self.dt = 0.02
        self.play_speed_ratio = 1.0
        self.speed_target = 1.0
        self._motion_done_sent = False
        self._skill_running = False
        if not self.cfg_ctrl.autoplay:
            self.play_speed_ratio = 0.0

        self._get_future_motion_targets()

    def _pack_motions(self, cfg_ctrl):
        """Merge the listed clips into one library so a skill key can select one."""
        names = list(getattr(cfg_ctrl, "motion_names", []) or [])
        if len(names) < 2:
            return
        import joblib

        from robojudo.config import ASSETS_DIR

        merged = {}
        for name in names:
            path = ASSETS_DIR / f"motions/{cfg_ctrl.robot}/phc/{name}.pkl"
            clip = joblib.load(path)
            merged[name] = next(iter(clip.values()))
        pack_path = ASSETS_DIR / f"motions/{cfg_ctrl.robot}/phc/kungfubot/_skill_pack.pkl"
        joblib.dump(merged, pack_path)
        cfg_ctrl.motion_name = "kungfubot/_skill_pack"

    def _take_motion_select(self, commands: list[str]) -> bool:
        selected = None
        for command in commands:
            if command.startswith("[MOTION_SELECT],"):
                selected = int(command.split(",")[1])
        if selected is None:
            return False
        self._motion_done_sent = False
        self._skill_running = False
        self.play_speed_ratio = 0.0
        self.load_motion(selected, block=True)
        self.reset()
        self._get_future_motion_targets()
        return True

    def post_step_callback(self, commands: list[str] | None = None):
        commands = list(commands or [])
        if self._take_motion_select(commands):
            return
        if not self.cfg_ctrl.autoplay:
            self._post_step_while_paused(commands)
            return
        self._get_future_motion_targets()
        super().post_step_callback(commands)

    def _post_step_while_paused(self, commands: list[str]):
        """Hold the clip until the kungfu policy is actually in control, then hand back at the end."""
        if "[POLICY_MIMIC]" in commands or "[MOTION_RESET]" in commands:
            self._motion_done_sent = False
            self._skill_running = False
            self.play_speed_ratio = 0.0
            self.reset()
            self._get_future_motion_targets()
            return

        if "[POLICY_LOCO]" in commands:
            self.play_speed_ratio = 0.0
            self._skill_running = False

        if "[MOTION_STEP]" in commands and not self._motion_done_sent:
            if not self._skill_running:
                self.reset()
                self.play_speed_ratio = 1.0
                self._skill_running = True
            if self.motion_time >= float(self.motion_length) - 1e-3:
                self.play_speed_ratio = 0.0
                self._skill_running = False
                self._motion_done_sent = True
                self.gui_commands.put("[POLICY_LOCO]")
                self._get_future_motion_targets()
                return
            self._get_future_motion_targets()
            MotionCtrl.post_step_callback(self, commands)
            return

        self._get_future_motion_targets()

    def _get_future_motion_targets(self):
        self.tar_obs_steps = torch.linspace(
            start=1,
            end=self.future_max_steps,
            steps=self.future_num_steps,
            device=self.device,
            dtype=torch.long,
        )
        num_steps = self.tar_obs_steps.numel()
        obs_motion_times = self.tar_obs_steps * self.dt + self.motion_time
        motion_ids = torch.zeros(num_steps, dtype=torch.int, device=self.device)
        motion_res = self.get_motion(motion_ids, obs_motion_times)

        root_rot = motion_res["root_rot"]
        root_pos = motion_res["root_pos"]
        root_vel = motion_res["root_vel"]
        root_ang_vel = motion_res["root_ang_vel"]
        dof_pos = motion_res["dof_pos"]
        ref_body_pos_extend = motion_res["rg_pos_t"]
        ref_body_rot_extend = motion_res["rg_rot_t"]

        flat_root_rot = root_rot.reshape(num_steps, 4)
        flat_root_vel = root_vel.reshape(num_steps, 3)
        flat_root_ang_vel = root_ang_vel.reshape(num_steps, 3)

        rpy = get_euler_xyz_in_tensor(flat_root_rot)
        roll_pitch = rpy[:, :2].reshape(num_steps, 2)

        root_vel = quat_rotate_inverse(flat_root_rot, flat_root_vel, w_last=True).view(num_steps, 3)
        root_ang_vel = quat_rotate_inverse(flat_root_rot, flat_root_ang_vel, w_last=True).view(num_steps, 3)

        robot_anchor_pos_w_repeat = ref_body_pos_extend[..., self.anchor_index, :][..., None, :].repeat(1, 27, 1)
        robot_anchor_quat_w_repeat = ref_body_rot_extend[..., self.anchor_index, :][..., None, :].repeat(1, 27, 1)
        local_ref_key_body_pos = quat_apply(
            quat_inverse(robot_anchor_quat_w_repeat, w_last=True),
            ref_body_pos_extend - robot_anchor_pos_w_repeat,
            w_last=True,
        )[..., self.key_body_id, :].reshape(num_steps, -1)

        self._obs_future_motion_root_height = root_pos[..., 2:3].flatten()
        self._obs_future_motion_roll_pitch = roll_pitch.flatten()
        self._obs_future_motion_base_lin_vel = root_vel.flatten()
        self._obs_future_motion_base_ang_vel = root_ang_vel.flatten()
        self._obs_future_motion_base_yaw_vel = root_ang_vel[..., 2:3].flatten()
        self._obs_future_motion_dof_pos = dof_pos.flatten()
        self._obs_future_motion_local_ref_key_body_pos = local_ref_key_body_pos.flatten()
        self._obs_next_step_ref_motion = torch.cat(
            [
                root_pos[0, 2:3].view(-1),
                roll_pitch[0, :].view(-1),
                root_vel[0, :].view(-1),
                root_ang_vel[0, 2:3].view(-1),
                dof_pos[0, :].view(-1),
                local_ref_key_body_pos[0, :].view(-1),
            ],
            dim=-1,
        )

        self._ref_frame_anchor = (root_pos[0], root_rot[0])

        # Hands
        if (hand_pose := motion_res.get("hand_pose", None)) is not None:
            self._hand_pose = hand_pose.cpu().numpy().squeeze().copy().reshape(2, -1)
        else:
            self._hand_pose = None

    def get_data(self):
        ref_frame_anchor_np = (
            self._ref_frame_anchor[0].cpu().numpy().copy(),
            self._ref_frame_anchor[1].cpu().numpy().copy(),
        )

        ctrl_data = {
            "future_motion_root_height": self._obs_future_motion_root_height.cpu().numpy().copy(),
            "future_motion_roll_pitch": self._obs_future_motion_roll_pitch.cpu().numpy().copy(),
            "future_motion_base_lin_vel": self._obs_future_motion_base_lin_vel.cpu().numpy().copy(),
            "future_motion_base_yaw_vel": self._obs_future_motion_base_yaw_vel.cpu().numpy().copy(),
            "future_motion_dof_pos": self._obs_future_motion_dof_pos.cpu().numpy().copy(),
            "next_step_ref_motion": self._obs_next_step_ref_motion.cpu().numpy().copy(),
            "ref_frame_anchor": ref_frame_anchor_np,
        }

        if self._hand_pose is not None:
            ctrl_data["hand_pose"] = self._hand_pose

        return ctrl_data


if __name__ == "__main__":
    from pprint import pprint

    from robojudo.config.g1.ctrl.g1_motion_ctrl_cfg import G1MotionKungfuBotCtrlCfg

    motion_ctrl = MotionKungfuBotCtrl(
        cfg_ctrl=G1MotionKungfuBotCtrlCfg(motion_name="singles/0-KIT_6_WalkInCounterClockwiseCircle05_1_poses"),
        env=None,
    )
    from robojudo.utils.progress import ProgressBar

    pbar = ProgressBar("test motion ctrl", total=-1)

    while True:
        ctrl_data = motion_ctrl.get_data()
        _, commands = motion_ctrl.process_triggers(ctrl_data)
        motion_ctrl.post_step_callback(commands)
        pprint(ctrl_data)
        pprint(commands)
        pbar.update(1)
        time.sleep(0.1)
