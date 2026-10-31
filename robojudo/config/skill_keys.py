"""Keyboard map shared by every locomotion / skill config.

`0` returns to locomotion. `1`, `2`, ... start skills in list order.
Only one digit is available per key, so skill 10 and later stay on `;` / `'`.
"""


def loco_mimic_keys(n_skills: int) -> dict[str, str]:
    keys = {"0": "[POLICY_LOCO]"}
    for index in range(min(n_skills, 9)):
        keys[str(index + 1)] = f"[POLICY_MIMIC],{index}"
    if n_skills > 9:
        keys[";"] = "[POLICY_SWITCH],NEXT"
        keys["'"] = "[POLICY_SWITCH],LAST"
    return keys


def multi_policy_keys(n_policies: int) -> dict[str, str]:
    """Policy 0 is locomotion. Policies 1, 2, ... are the other skills."""
    keys = {}
    for index in range(min(n_policies, 10)):
        keys[str(index)] = f"[POLICY_SWITCH],{index}"
    return keys
