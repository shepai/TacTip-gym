"""
Sim-to-real for the Cartesian rig: calibrate the real rig (github.com/shepai/Rig-controller,
MicroPython variant) and its TacTip against the simulated TacRig, so that what the real
sensor sees can be expressed as the simulated sensor's "tactile" input, and the same
experiment runs on both.

    markers.py      pins in the real TacTip's camera image: detection and tracking
    layout.py       which real pin is which simulated marker; pixels -> mm on the sensor
    rigs.py         SimRig and RealRig: one interface in mm, contact finding, run_probes()
    calibration.py  fit_calibration(), RigCalibration (JSON), RealToSim
    pose_map.py     PoseMap: simulated pose (mm) <-> real stepper position (steps)
    teleop.py       keyboard: pose_calibrate() (match both rigs by hand), follow()
    paired.py       paired sim / real samples from rig_dataset sessions, preprocessing
    mapping.py      the learned image mapping: sim and real autoencoders + latent maps;
                    train_mapping(), load_mapping() (sim in -> map -> real out, and back)
    real_pins.py    the real pins as coordinates (x, y from the image, z from depth)
    marker_map.py   simulated markers <-> real pins: train_marker_map(), load_marker_map()
    resim.py        re-run recorded trials in simulation (other readouts, z corrected)
    __main__.py     python -m tactip_mujoco_gym.sim2real real-setup | sim-capture |
                    real-capture | fit | experiment | compare | pose-calibrate | follow |
                    train-mapping | mapping-preview

The workflow is in __main__.py's docstring. Only numpy is needed to fit and apply a
calibration; the real rig adds OpenCV, pyserial and mpremote (as Rig-controller does), the
simulated one MuJoCo.
"""
from .calibration import RealToSim, RigCalibration, fit_calibration
from .layout import PolyMap, match_layouts, pick_by_motion
from .mapping import Sim2RealMapping, load_mapping, train_mapping
from .marker_map import MarkerMap, MarkerPairs, load_marker_map, train_marker_map
from .markers import MarkerDetector, MarkerTracker, match_rest
from .paired import PairedData, Preprocess
from .real_pins import depth_check, extract_pins
from .resim import resimulate
from .pose_map import PoseMap
from .rigs import (RealRig, SimRig, calibration_probes, connect_controller, contact_signal,
                   flat_block_stl, grid_probes, hinge_fit, jog, load_rig_setup,
                   measure_steps_per_mm, open_camera, run_probes, save_rig_setup)

__all__ = ["RealToSim", "RigCalibration", "fit_calibration", "PolyMap", "match_layouts",
           "pick_by_motion", "MarkerDetector", "MarkerTracker", "match_rest", "RealRig",
           "SimRig", "calibration_probes", "connect_controller", "contact_signal", "flat_block_stl", "grid_probes",
           "hinge_fit", "jog", "load_rig_setup", "measure_steps_per_mm", "open_camera", "run_probes",
           "save_rig_setup", "PoseMap", "PairedData", "Preprocess", "Sim2RealMapping",
           "load_mapping", "train_mapping", "MarkerMap", "MarkerPairs", "load_marker_map",
           "train_marker_map", "depth_check", "extract_pins", "resimulate"]
