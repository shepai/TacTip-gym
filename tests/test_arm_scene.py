"""Taking the built-in TacTip out of the arm scenes (plain XML, no MuJoCo needed to check)."""
import os
import xml.etree.ElementTree as ET

import pytest

from conftest import ASSETS, needs_arm

pytest.importorskip("mujoco")
from tactip_mujoco_gym import sensor_tools as st  # noqa: E402

SCENES = ["tactip_arm.xml", "peg_worldbody_tx.xml", "tactip_arm_shapes.xml",
          "hill_worldbody_tx.xml", "edge_worldbody_tx.xml"]


def _names(root, tag):
    return {e.get("name") for e in root.iter(tag) if e.get("name")}


@pytest.mark.parametrize("scene", SCENES)
def test_builtin_tactip_removed_cleanly(scene):
    path = os.path.join(ASSETS, scene)
    kids = st.child_bodies(path, "sensor_root")
    assert kids == ["flexible_structure"]
    xml, removed = st.scene_without_bodies(path, kids)
    root = ET.fromstring(xml)
    assert not any(n.startswith("s_c") for n in _names(root, "site"))
    assert {"ee_site"} <= _names(root, "site") and {"handcam"} <= _names(root, "camera")
    assert _names(root, "body") >= {"link06", "sensor_root"}
    defined = {k: _names(root, k) for k in ("body", "joint", "site", "geom", "tendon")}
    for section in ("tendon", "equality", "contact", "sensor", "actuator"):
        for sec in root.findall(section):
            for child in sec:
                for kind, name in st._refs(child):
                    if kind in defined:
                        assert name in defined[kind], f"{section}/{child.tag} -> missing {kind} {name}"
    assert len(removed["tendon"]) > 100 and len(removed["equality"]) > 50
    assert len(list(root.find("actuator"))) == 6                      # the arm's own


@needs_arm
def test_stripped_scene_compiles():
    import mujoco
    path = os.path.join(ASSETS, "peg_worldbody_tx.xml")
    xml, _ = st.scene_without_bodies(path, st.child_bodies(path, "sensor_root"))
    m = mujoco.MjSpec.from_string(xml).compile()
    assert m.nu == 6 and m.ntendon == 0
