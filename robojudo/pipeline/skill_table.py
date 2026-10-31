"""Print the keyboard skill map once the pipeline has loaded its policies."""

from pathlib import Path


def skill_label(policy) -> str:
    cfg = policy.cfg_policy
    title = getattr(cfg, "skill_title", "") or ""
    if title:
        return title
    name = getattr(cfg, "policy_name", "") or ""
    relative = getattr(cfg, "relative_path", "") or ""
    stem = Path(relative).stem
    short = name.split("/")[-1]
    if stem and not stem.startswith("model") and stem != short:
        return stem
    return short or stem or cfg.policy_type


def print_skill_table(ctrl_manager, policies, *, mimic: bool) -> None:
    controllers = ctrl_manager.controllers
    if "KeyboardCtrl" not in controllers:
        return
    triggers = controllers["KeyboardCtrl"].inst.triggers

    rows: list[tuple[str, str]] = []
    for key in sorted(triggers, key=lambda item: (0, int(item)) if item.isdigit() else (1, item)):
        command = triggers[key]
        label = _command_label(key, command, policies, mimic=mimic)
        if label is not None:
            rows.append((key, label))
    if not rows:
        return

    width = max(len(key) for key, _ in rows)
    lines = ["", "key  skill"]
    for key, label in rows:
        lines.append(f"{key:<{width}}  {label}")
    print("\n".join(lines), flush=True)


def _command_label(key: str, command: str, policies, *, mimic: bool) -> str | None:
    if command == "[SHOW_SKILLS]":
        return "show this table"
    if command == "[SIM_REBORN]":
        if key != "r":
            return None
        return "reset simulation"
    if command == "[POLICY_LOCO]":
        return "locomotion"
    if command == "[POLICY_TOGGLE]":
        return "next policy"
    if command == "[POLICY_SWITCH],NEXT":
        return "next skill"
    if command == "[POLICY_SWITCH],LAST":
        return "previous skill"

    if command == "[POLICY_MIMIC]" or command.startswith("[POLICY_MIMIC],"):
        index = 0
        parts = command.split(",")
        if len(parts) == 2 and parts[1].isdigit():
            index = int(parts[1])
        policy_id = index + 1 if mimic else index
        if 0 <= policy_id < len(policies):
            return skill_label(policies[policy_id])
        return None

    if command.startswith("[POLICY_SWITCH],"):
        part = command.split(",")[1]
        if part.isdigit():
            policy_id = int(part)
            if 0 <= policy_id < len(policies):
                if policy_id == 0:
                    return "locomotion"
                return skill_label(policies[policy_id])
    return None
