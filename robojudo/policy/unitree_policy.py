import time

import numpy as np

from robojudo.environment.utils.mujoco_viz import MujocoVisualizer
from robojudo.policy import Policy, policy_registry
from robojudo.policy.policy_cfgs import UnitreePolicyCfg, UnitreeWoGaitPolicyCfg
from robojudo.utils.util_func import command_remap, get_gravity_orientation


@policy_registry.register
class UnitreePolicy(Policy):
    cfg_policy: UnitreePolicyCfg

    def __init__(self, cfg_policy, device):
        super().__init__(cfg_policy=cfg_policy, device=device)

        self.obs_scales = self.cfg_policy.obs_scales
        self.max_cmd = self.cfg_policy.max_cmd
        self.commands_map = self.cfg_policy.commands_map

        self.reset()

    def _reset_kb_cmd(self):
        self._kb_held: set[str] = set()
        self._kb_press_t: dict[str, float] = {}
        self._kb_pulse_until: dict[str, float] = {}
        self._kb_smooth = np.zeros(3, dtype=np.float32)
        self._last_kb_cmd: tuple[float, ...] | None = None

    def reset(self):
        self.timestep: int = 0
        self._reset_kb_cmd()

        self._init_history(np.zeros(self.history_obs_size))

    def post_step_callback(self, commands=None):
        self.timestep += 1

    def _get_phase(self):
        cycle_time = 0.8
        phase = self.timestep * self.dt / cycle_time
        return phase

    def _get_commands(self, ctrl_data):
        commands = np.zeros(3)
        for key in ctrl_data.keys():
            if key in ["JoystickCtrl", "UnitreeCtrl"]:
                axes = ctrl_data[key]["axes"]
                lx, ly, rx, ry = axes["LeftX"], axes["LeftY"], axes["RightX"], axes["RightY"]

                commands[0] = command_remap(ly, self.commands_map[0])
                commands[1] = command_remap(lx, self.commands_map[1])
                commands[2] = command_remap(rx, self.commands_map[2])
                break
            if key in ["KeyboardCtrl"]:
                # A tap arrives as down+up in the same millisecond, so the key is
                # already released before the next control step. Keep that command
                # alive briefly, and refresh it while the key repeats.
                now = time.monotonic()
                for event in ctrl_data[key]["keyboard_event"]:
                    name = event["name"]
                    if event["type"] != "keyboard" or name not in ("w", "a", "s", "d", "q", "e"):
                        continue
                    if event["pressed"]:
                        self._kb_press_t[name] = float(event["timestamp"])
                        self._kb_held.add(name)
                    elif float(event["timestamp"]) - self._kb_press_t.get(name, 0.0) < 0.05:
                        self._kb_held.discard(name)
                        self._kb_pulse_until[name] = now + 0.45
                    else:
                        self._kb_held.discard(name)
                        self._kb_pulse_until.pop(name, None)
                active = set(self._kb_held)
                for name, until in list(self._kb_pulse_until.items()):
                    if until >= now:
                        active.add(name)
                    else:
                        self._kb_pulse_until.pop(name, None)
                # Full stick is 1.0 (~0.8 m/s forward). Hold the key and the command climbs there.
                stick = 1.0
                target = np.zeros(3, dtype=np.float32)
                if "w" in active:
                    target[0] = command_remap(stick, self.commands_map[0])
                elif "s" in active:
                    target[0] = command_remap(-stick, self.commands_map[0])
                if "a" in active:
                    target[1] = command_remap(-stick, self.commands_map[1])
                elif "d" in active:
                    target[1] = command_remap(stick, self.commands_map[1])
                if "q" in active:
                    target[2] = command_remap(-stick, self.commands_map[2])
                elif "e" in active:
                    target[2] = command_remap(stick, self.commands_map[2])
                # About 3 s from zero to full, and the same on the way back down.
                step = np.float32(self.dt / 3.0)
                self._kb_smooth += np.clip(target - self._kb_smooth, -step, step)
                commands = self._kb_smooth.astype(np.float32, copy=True)
                cmd = tuple(round(float(c), 1) for c in commands)
                if cmd != self._last_kb_cmd:
                    self._last_kb_cmd = cmd
                    print(f"keyboard cmd {list(cmd)} held={sorted(active)}", flush=True)
                break
        return commands

    def get_observation(self, env_data, ctrl_data):
        phase = self._get_phase()
        commands = self._get_commands(ctrl_data)

        sin_pos = [np.sin(2 * np.pi * phase)]
        cos_pos = [np.cos(2 * np.pi * phase)]

        gravity_orientation = get_gravity_orientation(env_data.base_quat)
        obs = np.concatenate(
            [
                env_data.base_ang_vel * self.obs_scales.ang_vel,
                gravity_orientation,
                commands * self.obs_scales.command * self.max_cmd,
                env_data.dof_pos - self.default_dof_pos,
                env_data.dof_vel * self.obs_scales.dof_vel,
                self.last_action,
                sin_pos,
                cos_pos,
            ]
        )

        extras = {
            "phase": phase,
            "commands": commands,
        }
        return obs, extras

    def debug_viz(self, visualizer: MujocoVisualizer, env_data, ctrl_data, extras):
        base_pos = env_data["base_pos"]
        base_quat = env_data["base_quat"]
        command_x = extras["commands"][0]
        command_y = extras["commands"][1]
        command_yaw = extras["commands"][2]

        visualizer.draw_arrow(
            base_pos,
            base_quat,
            [command_x, 0, 0],
            color=[1, 0, 0, 1],
            scale=2,
            horizontal_only=True,
            id=0,
        )
        visualizer.draw_arrow(
            base_pos,
            base_quat,
            [0, command_y, 0],
            color=[0, 1, 0, 1],
            scale=2,
            horizontal_only=True,
            id=1,
        )
        visualizer.draw_arrow(
            base_pos + np.array([0.0, 0, 0.6]),
            base_quat,
            [0, command_yaw, 0],
            color=[1, 1, 1, 1],
            scale=2,
            horizontal_only=True,
            id=2,
        )


@policy_registry.register
class UnitreeWoGaitPolicy(UnitreePolicy):
    cfg_policy: UnitreeWoGaitPolicyCfg

    def __init__(self, cfg_policy, device):
        super().__init__(cfg_policy=cfg_policy, device=device)

    def reset(self):
        self.timestep: int = 0
        self._reset_kb_cmd()

        history_obs_dims = self.cfg_policy.history_obs_dims
        default_history = [np.zeros(dim, dtype=np.float32) for dim in history_obs_dims.values()]
        self._init_history(default_history)

    def get_observation(self, env_data, ctrl_data):
        commands = self._get_commands(ctrl_data)

        gravity_orientation = get_gravity_orientation(env_data.base_quat)
        obs_current = [
            env_data.base_ang_vel * self.obs_scales.ang_vel,
            gravity_orientation * self.obs_scales.gravity,
            commands * self.obs_scales.command * self.max_cmd,
            (env_data.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            env_data.dof_vel * self.obs_scales.dof_vel,
            self.last_action,
        ]
        self.history_buf.append(obs_current)

        history_list = [np.concatenate(items, axis=0) for items in zip(*self.history_buf, strict=True)]
        obs = np.concatenate(history_list, axis=0)

        extras = {
            "commands": commands,
        }
        return obs, extras
