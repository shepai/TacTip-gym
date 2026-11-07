"""
TacTip-gym: a MuJoCo soft-body TacTip and a Gymnasium benchmark for testing tactile, visual
and visuo-tactile sensing on several robots.

    import gymnasium as gym
    import tactip_mujoco_gym                          # registers the tasks

    env = gym.make("TacArm-peg", obs_modes=["tactile", "vision"], obs_format="dict")

The pieces a researcher picks (each documented in its module):
    task / robot      envs/__init__.py   TASKS, list_tasks()
    sensor hardware   sensors/models.py  sensor_xml=..., register_sensor_model()
    sensor readouts   sensors/base.py    sensors={...}, register_sensor(), Sensor
    inputs            envs/base.py       obs_modes=..., obs_format="flat" | "dict"
    fusion / policy   vt_encoders.py     RL_BASE.run(..., encoders=...)
    experiments       experiment.py      python -m tactip_mujoco_gym train <config.yaml>
"""
from . import envs
from .envs import TASKS, list_tasks
from . import sensors
from .sensors import (Sensor, SensorUnavailable, list_sensor_models, list_sensors,
                      make_sensor, register_sensor, register_sensor_model)

__all__ = ["TASKS", "list_tasks", "sensors", "Sensor", "SensorUnavailable", "register_sensor",
           "make_sensor", "list_sensors", "register_sensor_model", "list_sensor_models"]
