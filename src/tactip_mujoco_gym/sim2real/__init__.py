"""
Sim-to-real for the Cartesian rig: calibrate the real rig (github.com/shepai/Rig-controller,
MicroPython variant) and its TacTip against the simulated TacRig, so that what the real
sensor sees can be expressed as the simulated sensor's "tactile" input, and the same
experiment runs on both.

    markers.py      pins in the real TacTip's camera image: detection and tracking
    layout.py       which real pin is which simulated marker; pixels -> mm on the sensor
    rigs.py         SimRig and RealRig: one interface in mm, contact finding, run_probes()
    calibration.py  fit_calibration(), RigCalibration (JSON), RealToSim
    __main__.py     python -m tactip_mujoco_gym.sim2real sim-capture | real-capture | fit |
                    experiment | compare

The workflow is in __main__.py's docstring. Only numpy is needed to fit and apply a
calibration; the real rig adds OpenCV, pyserial and mpremote (as Rig-controller does), the
simulated one MuJoCo.
"""
from .calibration import RealToSim, RigCalibration, fit_calibration
from .layout import PolyMap, match_layouts, pick_by_motion
from .markers import MarkerDetector, MarkerTracker, match_rest
from .rigs import (RealRig, SimRig, calibration_probes, connect_controller, contact_signal,
                   flat_block_stl, grid_probes, hinge_fit, jog, measure_steps_per_mm,
                   open_camera, run_probes)

__all__ = ["RealToSim", "RigCalibration", "fit_calibration", "PolyMap", "match_layouts",
           "pick_by_motion", "MarkerDetector", "MarkerTracker", "match_rest", "RealRig",
           "SimRig", "calibration_probes", "connect_controller", "contact_signal", "flat_block_stl", "grid_probes",
           "hinge_fit", "jog", "measure_steps_per_mm", "open_camera", "run_probes"]
