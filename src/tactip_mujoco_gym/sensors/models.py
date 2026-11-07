"""
Sensor hardware: the MuJoCo model of the sensor that is mounted on the robot.

Every robot takes `sensor_xml=` (a registered name, a path to a standalone MJCF file, or an
MjSpec). The file holds just the sensor, with its root body first in <worldbody>:

    <mujoco>
      <asset> ... meshes / materials the sensor uses ... </asset>
      <worldbody>
        <body name="my_sensor"> ... </body>
      </worldbody>
      <tendon> / <equality> / <contact> ... anything between its parts ...
    </mujoco>

It is attached with a name prefix per mount (FL_, left_, ...; see sensor_tools.attach_sensor),
turned so its tip points the way the original TacTip's did and moved so the tips coincide.
The tip is found, in order of preference, from:
    1. a site named "sensor_tip" in the file: its position is the tip, its z axis points out
       of the sensing surface   <site name="sensor_tip" pos="0 0 0.03" zaxis="0 0 1"/>
    2. TacTip-style marker sites (s_c*): a dome is fitted to them
so a sensor without markers needs the "sensor_tip" site.

    register_sensor_model("my_gelsight", "path/to/gelsight.xml")
    gym.make("TacArm-peg", sensor_xml="my_gelsight")

"tactip" is the default: assets/sensor/tactip.xml if present, else the TacTip extracted from
assets/tactip_arm.xml (python -m tactip_mujoco_gym.sensor_tools extract ... writes the file).
"""
import os
from importlib.resources import files

import mujoco

_MODELS = {}
_CACHE = {}


def _asset(*parts):
    return str(files("tactip_mujoco_gym").joinpath(*parts))


def register_sensor_model(name, source, description=""):
    """source: path to a standalone sensor MJCF, or a callable returning an MjSpec / path."""
    _MODELS[name] = (source, description)


def list_sensor_models():
    return {k: d for k, (_, d) in sorted(_MODELS.items())}


def default_tactip(source=("assets", "tactip_arm.xml"), root="auto",
                   standalone=("assets", "sensor", "tactip.xml"), verbose=True):
    """The TacTip as a standalone MjSpec: `standalone` if that file exists, else extracted
    from the `source` scene (cached per process)."""
    path = _asset(*standalone) if isinstance(standalone, (tuple, list)) else str(standalone)
    if os.path.exists(path):
        return mujoco.MjSpec.from_file(path)
    src = _asset(*source) if isinstance(source, (tuple, list)) else str(source)
    key = ("extract", src, root)
    if key not in _CACHE:
        from tactip_mujoco_gym.sensor_tools import extract_sensor
        if verbose:
            print(f"[tactip_mujoco_gym] {path} not found: using the sensor in {src} "
                  f"(`python -m tactip_mujoco_gym.sensor_tools extract {src} --out {path}` "
                  "writes the file)")
        _CACHE[key] = extract_sensor(src, root)
    return _CACHE[key].copy()


register_sensor_model("tactip", default_tactip,
                      "the TacTip soft-body sensor (71 markers, in-sensor camera)")


def resolve_sensor_model(model=None, **default_kwargs):
    """MjSpec for `model`: None or "tactip" -> the default TacTip (default_kwargs are passed
    to default_tactip), a registered name, a path to an MJCF file, or an MjSpec (copied)."""
    if model is None:
        return default_tactip(**default_kwargs)
    if isinstance(model, mujoco.MjSpec):
        return model.copy()
    model = str(model)
    if model in _MODELS:
        source = _MODELS[model][0]
        if source is default_tactip:
            return default_tactip(**default_kwargs)
        out = source() if callable(source) else source
        return out.copy() if isinstance(out, mujoco.MjSpec) else _spec_from_path(out)
    if os.path.exists(model):
        return _spec_from_path(model)
    raise FileNotFoundError(f"sensor model {model!r} is neither a registered name "
                            f"({sorted(_MODELS)}) nor an existing file")


def _spec_from_path(path):
    from tactip_mujoco_gym.sensor_tools import _as_spec
    return _as_spec(path)                   # asset paths made absolute, so it attaches anywhere


def model_label(model):
    """Short text for logs."""
    if model is None:
        return "tactip (default)"
    if isinstance(model, mujoco.MjSpec):
        return f"MjSpec {model.modelname!r}"
    return str(model)
