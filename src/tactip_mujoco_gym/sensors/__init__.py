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
    contact_depth    "tactile_image"   contact depth image (Tactile Gym 2.0) or GelSight RGB
    contact_map      "tactile"         the contact depth on a coarse grid (mm)
    presstip         "tactile"         PressTip pad forces, piezo vibration, acceleration
    whiskers         "tactile"         whisker bend direction, contact and vibration

Built-in hardware (sensor_xml=...): "tactip" (the default), "presstip", "gelsight",
"tactip_depth" and "whiskers" (sensors/hardware.py). Hardware used by name brings the
readouts that suit it, e.g. sensor_xml="whiskers" makes "tactile" the whiskers readout.
"""
from .base import (Mount, Sensor, SensorUnavailable, frame_pose, list_sensors, make_sensor,
                   register_sensor, sensor_class, sensor_type_name, subtree_bodies, subtree_geoms)
from .camera import Camera
from .depth import ContactDepth, ContactMap
from .models import (list_sensor_models, register_sensor_model, resolve_sensor_model,
                     sensor_model_readouts)
from .presstip import PressTip
from .tactip import TacTipCamera, TacTipMarkers
from .taxels import TaxelEvents
from .whiskers import Whiskers

__all__ = [
    "Sensor", "SensorUnavailable", "Mount", "register_sensor", "make_sensor", "sensor_class",
    "sensor_type_name", "list_sensors", "frame_pose", "subtree_bodies", "subtree_geoms",
    "TacTipMarkers", "TacTipCamera", "Camera", "TaxelEvents", "ContactDepth", "ContactMap",
    "PressTip", "Whiskers",
    "register_sensor_model", "resolve_sensor_model", "list_sensor_models", "sensor_model_readouts",
]
