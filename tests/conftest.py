"""Shared helpers. Tests that need MuJoCo skip without it; tests that build a task skip when
its meshes / robot models aren't in assets/ (they are not in every copy of the repo)."""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)                       # so `examples.custom_sensor` imports
ASSETS = os.path.join(ROOT, "src", "tactip_mujoco_gym", "assets")


def have_arm_meshes():
    return os.path.exists(os.path.join(ASSETS, "robotarm", "assets", "z1_Link00.stl"))


needs_arm = pytest.mark.skipif(not have_arm_meshes(), reason="Z1 meshes (assets/robotarm/assets/*.stl) missing")


def robot_available(cls):
    try:
        cls.find_robot()
        return True
    except Exception:
        return False
