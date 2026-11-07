"""
Sensors: what is mounted on the robot (hardware) and how it is read (readouts).

    hardware   sensor_xml=...   the MuJoCo bodies of the sensor (default: the TacTip);
                                see sensors/models.py
    readouts   sensors={...}    observation inputs computed from the simulation; see
                                sensors/base.py for writing your own

Built-in readouts (python -m tactip_mujoco_gym list shows the current list):

    tactip_markers   "tactile"         TacTip marker displacements (+ change), every mount
    tactip_camera    "tactile_image"   image from the camera inside each TacTip
    camera           "vision"          image from a scene / robot camera
    taxel_events     "events"          event counts from tx_* touch grids on the objects
"""
from .base import (Mount, Sensor, SensorUnavailable, frame_pose, list_sensors, make_sensor,
                   register_sensor, sensor_class, sensor_type_name, subtree_bodies, subtree_geoms)
from .camera import Camera
from .models import list_sensor_models, register_sensor_model, resolve_sensor_model
from .tactip import TacTipCamera, TacTipMarkers
from .taxels import TaxelEvents

__all__ = [
    "Sensor", "SensorUnavailable", "Mount", "register_sensor", "make_sensor", "sensor_class",
    "sensor_type_name", "list_sensors", "frame_pose", "subtree_bodies", "subtree_geoms",
    "TacTipMarkers", "TacTipCamera", "Camera", "TaxelEvents",
    "register_sensor_model", "resolve_sensor_model", "list_sensor_models",
]
