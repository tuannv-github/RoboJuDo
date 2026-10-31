import logging
import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable

import numpy as np
import torch
from tqdm import tqdm

from robojudo.tools.tool_cfgs import DoFConfig
from robojudo.utils.util_func import command_remap

from .policy_cfgs import PolicyCfg

logger = logging.getLogger(__name__)


class Policy(ABC):
    def __init__(self, cfg_policy: PolicyCfg, device: str = "cpu"):
        self.cfg_policy = cfg_policy
        self.device = device

        self.freq = self.cfg_policy.freq
        self.dt = 1.0 / self.freq

        self.cfg_obs_dof: DoFConfig = self.cfg_policy.obs_dof
        self.cfg_action_dof: DoFConfig = self.cfg_policy.action_dof

        self.num_dofs = self.cfg_obs_dof.num_dofs
        self.num_actions = self.cfg_action_dof.num_dofs

        self.default_dof_pos = np.asarray(self.cfg_obs_dof.default_pos)
        self.default_pos = np.asarray(self.cfg_action_dof.default_pos)  # TODO: remove

        # TODO: autoload cfg
        if self.cfg_policy.disable_autoload:
            # self.model: torch.nn.Module | None = None # type: ignore
            pass
        else:
            policy_file = self.cfg_policy.policy_file
            logger.debug(f"Loading jit from {policy_file}...")
            self.model = torch.jit.load(policy_file, map_location=self.device)

        self.action_scale = self.cfg_policy.action_scale
        self.action_clip = self.cfg_policy.action_clip
        self.action_beta = self.cfg_policy.action_beta

        self.last_action = np.zeros(self.num_actions)

        self.history_length = self.cfg_policy.history_length
        self.history_obs_size = self.cfg_policy.history_obs_size
        self._reset_keyboard()

    def _reset_keyboard(self):
        self._kb_held: set[str] = set()
        self._kb_press_t: dict[str, float] = {}
        self._kb_pulse_until: dict[str, float] = {}
        # Signed stick in [-1, 1]: forward (w/s), lateral (a -, d +), yaw (q -, e +).
        self._kb_smooth = np.zeros(3, dtype=np.float32)
        self._last_kb_cmd: tuple[float, ...] | None = None

    def _keyboard_command(
        self,
        events,
        commands_map,
        on_press: Callable[[str], None] | None = None,
    ) -> np.ndarray:
        """Hold-to-move command from KeyboardCtrl events.

        A tap arrives as down+up in the same millisecond, so the key is already
        released before the next control step. That press stays active briefly.
        The stick ramps to full in about 3 seconds, then ``commands_map`` scales
        each axis. ``on_press`` receives other keys on the press edge.
        """
        now = time.monotonic()
        move_keys = ("w", "a", "s", "d", "q", "e")
        for event in events:
            name = event["name"]
            if event.get("type") != "keyboard":
                continue
            if name in move_keys:
                if event["pressed"]:
                    self._kb_press_t[name] = float(event["timestamp"])
                    self._kb_held.add(name)
                elif float(event["timestamp"]) - self._kb_press_t.get(name, 0.0) < 0.05:
                    self._kb_held.discard(name)
                    self._kb_pulse_until[name] = now + 0.45
                else:
                    self._kb_held.discard(name)
                    self._kb_pulse_until.pop(name, None)
            elif event["pressed"] and on_press is not None:
                on_press(name)

        active = set(self._kb_held)
        for name, until in list(self._kb_pulse_until.items()):
            if until >= now:
                active.add(name)
            else:
                self._kb_pulse_until.pop(name, None)

        stick = np.zeros(3, dtype=np.float32)
        if "w" in active:
            stick[0] = 1.0
        elif "s" in active:
            stick[0] = -1.0
        if "a" in active:
            stick[1] = -1.0
        elif "d" in active:
            stick[1] = 1.0
        if "q" in active:
            stick[2] = -1.0
        elif "e" in active:
            stick[2] = 1.0

        step = np.float32(self.dt / 3.0)
        self._kb_smooth += np.clip(stick - self._kb_smooth, -step, step)
        command = np.array(
            [float(command_remap(float(self._kb_smooth[i]), commands_map[i])) for i in range(3)],
            dtype=np.float32,
        )
        rounded = tuple(round(float(c), 1) for c in command)
        if rounded != self._last_kb_cmd:
            self._last_kb_cmd = rounded
            tqdm.write(f"keyboard cmd {list(rounded)} held={sorted(active)}")
        return command

    def _init_history(self, default_history: np.ndarray | torch.Tensor | list):
        logger.debug(f"Initializing history buffer as {self.history_length} x {len(default_history)}")
        self.history_buf = deque(maxlen=self.history_length)
        for _ in range(self.history_length):
            self.history_buf.append(default_history)

    @abstractmethod
    def reset(self):
        # self.last_action = np.zeros(self.num_actions) # TODO
        raise NotImplementedError

    @abstractmethod
    def post_step_callback(self, commands: list[str] | None = None):
        raise NotImplementedError

    @abstractmethod
    def get_observation(self, env_data, ctrl_data) -> tuple[np.ndarray, dict]:
        raise NotImplementedError

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        obs_tensor = torch.from_numpy(obs).unsqueeze(0).float().to(self.device)
        with torch.no_grad(), torch.backends.cudnn.flags(enabled=False):
            # Scripted LSTMs segfault in cuDNN when their weights are not one contiguous chunk.
            actions_tensor = self.model(obs_tensor).cpu()

        actions = actions_tensor.numpy().squeeze()
        actions = (1 - self.action_beta) * self.last_action + self.action_beta * actions

        self.last_action = actions.copy()  # TODO

        processed_actions = actions
        if self.action_clip is not None:
            processed_actions = np.clip(processed_actions, -self.action_clip, self.action_clip)

        processed_actions = processed_actions * self.action_scale
        return processed_actions

    def get_init_dof_pos(self) -> np.ndarray:
        """
        Return the initial dof pos for the policy, used for robot preparation.
        For motion policies, this should return next/first frame of the reference motion.
        """
        return self.default_pos.copy()

    def debug_viz(self, visualizer, env_data, ctrl_data, extras):
        # for debug draw
        return
