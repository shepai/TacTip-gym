from importlib.resources import files

xml_path = files("tactip_mujoco_gym").joinpath(
    "assets",
    "robotarm",
    "scene.xml"
)

print(xml_path)