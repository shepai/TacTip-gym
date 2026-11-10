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

    register_sensor_model("my_gelsight", "path/to/gelsight.xml",
                          readouts={"tactile_image": "contact_depth"})
    gym.make("TacArm-peg", sensor_xml="my_gelsight")

`readouts` are the inputs the hardware brings when it is used by name: they replace the
robot's defaults for those keys, and sensors={...} still changes them (a dict without "type"
tweaks them: sensors={"tactile_image": {"mode": "depth"}}).

Built in: "tactip" (the default), and from sensors/hardware.py "presstip", "gelsight",
"tactip_depth" and "whiskers" (`python -m tactip_mujoco_gym list` shows them).

"tactip" is the default: assets/sensor/tactip.xml if present. Otherwise the TacTip is
extracted from assets/tactip_arm.xml, written to that file (if it loads back into the same
model; delete the file to extract again after changing the sensor in tactip_arm.xml), and
read from there from then on. `python -m tactip_mujoco_gym.sensor_tools extract ...` writes
it by hand.
"""
import os
from importlib.resources import files

import mujoco
import numpy as np

_MODELS = {}
_CACHE = {}


def _asset(*parts):
    return str(files("tactip_mujoco_gym").joinpath(*parts))


def register_sensor_model(name, source, description="", readouts=None):
    """source: path to a standalone sensor MJCF, or a callable returning an MjSpec / path.
    readouts: {input key: readout spec} the hardware brings when used by name, e.g.
    {"tactile": "presstip"} (see the module docstring)."""
    _MODELS[name] = (source, description, dict(readouts or {}))


def list_sensor_models():
    return {k: d for k, (_, d, _) in sorted(_MODELS.items())}


def sensor_model_readouts(model):
    """The readouts registered with a sensor model ({} for files, MjSpecs and unknown names)."""
    if isinstance(model, str) and model in _MODELS:
        return {k: (dict(v) if isinstance(v, dict) else v) for k, v in _MODELS[model][2].items()}
    return {}


def default_tactip(source=("assets", "tactip_arm.xml"), root="auto",
                   standalone=("assets", "sensor", "tactip.xml"), verbose=True):
    """The TacTip as a standalone MjSpec: `standalone` if that file exists, else extracted
    from the `source` scene and written to `standalone` (cached per process). Reading the
    file avoids extracting in every process, and MuJoCo's attach notices with it."""
    path = _asset(*standalone) if isinstance(standalone, (tuple, list)) else str(standalone)
    if os.path.exists(path):
        return mujoco.MjSpec.from_file(path)
    src = _asset(*source) if isinstance(source, (tuple, list)) else str(source)
    key = ("extract", src, root, path)
    if key not in _CACHE:
        from tactip_mujoco_gym.sensor_tools import extract_sensor
        spec = extract_sensor(src, root)
        written = _write_standalone(spec, src, root, path)
        if verbose:
            print(f"[tactip_mujoco_gym] {path} not found: " + (
                f"wrote the TacTip from {src} there (used from now on; delete it to extract "
                "again after changing the sensor in that scene)" if written else
                f"using the sensor in {src} for this process (`python -m tactip_mujoco_gym."
                f"sensor_tools extract {src} --out {path}` writes the file)"))
        _CACHE[key] = mujoco.MjSpec.from_file(path) if written else spec
    return _CACHE[key].copy()


_SAME_ARRAYS = ("body_parentid", "body_mass", "body_pos", "body_quat", "body_inertia",
                "jnt_type", "jnt_range", "jnt_stiffness", "qpos_spring", "dof_damping",
                "dof_frictionloss", "dof_armature", "geom_type", "geom_size", "geom_pos",
                "geom_quat", "geom_contype", "geom_conaffinity", "geom_friction", "geom_solref",
                "geom_solimp", "geom_group", "site_pos", "cam_pos", "tendon_stiffness",
                "tendon_damping", "tendon_width", "tendon_solref_lim", "tendon_solimp_lim",
                "eq_type", "eq_data", "eq_solref", "eq_solimp")


def _same_model(a, b):
    """Whether two compiled models of the sensor are the same (sizes and the parameters that
    shape its behaviour)."""
    for k in ("nbody", "njnt", "ngeom", "nsite", "ncam", "ntendon", "neq", "nq", "nv", "nu"):
        if getattr(a, k) != getattr(b, k):
            return False
    for k in _SAME_ARRAYS:
        x, y = getattr(a, k, None), getattr(b, k, None)
        if x is None or y is None:
            continue
        if np.shape(x) != np.shape(y):
            return False
        scale = float(np.abs(x).max()) if np.size(x) else 0.0      # (XML keeps ~7 digits)
        if not np.allclose(x, y, rtol=1e-5, atol=1e-6 * scale):
            return False
    return True


def _write_standalone(spec, src, root, path):
    """Write the extracted TacTip to `path` (atomically: several processes may do this at
    once on a first run) if it loads back into the same model. Returns whether it did."""
    from tactip_mujoco_gym.sensor_tools import compile_spec, save_spec
    base = path[:-4] if path.endswith(".xml") else path
    tmp = f"{base}.{os.getpid()}.tmp.xml"
    try:
        save_spec(spec.copy(), tmp)
        if not _same_model(compile_spec(spec.copy()), compile_spec(mujoco.MjSpec.from_file(tmp))):
            print(f"[tactip_mujoco_gym] the TacTip written from {src} doesn't load back the same: "
                  "not keeping the file")
            return False
        os.replace(tmp, path)
        return True
    except Exception as err:                      # read-only install, odd model: carry on in memory
        if os.path.exists(path):                  # another process just wrote it
            return True
        print(f"[tactip_mujoco_gym] couldn't write {path} ({err}); extracting in each process")
        return False
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


register_sensor_model("tactip", default_tactip,
                      "the TacTip soft-body sensor (71 markers, in-sensor camera)")


def _register_built_in():
    from . import hardware
    for name in hardware.BUILDERS:
        build = getattr(hardware, name if name != "whiskers" else "whisker_array")
        register_sensor_model(name, build, hardware.DESCRIPTIONS[name], hardware.READOUTS[name])


_register_built_in()


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
