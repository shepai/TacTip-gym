"""
Drive the simulated rig with the keyboard; the real rig follows in the background.

First make the pose calibration (once, both rigs live, matched by hand):

    python -m tactip_mujoco_gym.sim2real pose-calibrate --port /dev/ttyACM0 --camera 1 \
        --out calib/pose_calibration.json

Then:

    python examples/sim2real/follow_demo.py --port /dev/ttyACM0 --camera 1

(Rig-controller installed as a package: pip install -e path/to/Rig-controller.)

In the MuJoCo viewer (or the "real rig" window, which shows the TacTip camera):
    up / down     +x / -x            PgUp / PgDn   z up / down
    right / left  +y / -y            1 .. 5        step size ("real rig" window)
    z             reset the simulated rig          h   home the real rig
    q / Esc       quit ("real rig" window; the real rig homes before closing)

The real rig homes first (z up, then x and y), then goes wherever the simulated one is. It is
kept within --margin mm (x, y) of the poses the calibration was recorded at, above home, and
within the simulated rig's own z travel. On macOS run this with `mjpython`.
"""
import argparse

from tactip_mujoco_gym.sim2real import PoseMap, RealRig, SimRig, connect_controller, open_camera
from tactip_mujoco_gym.sim2real.rigs import flat_block_stl
from tactip_mujoco_gym.sim2real.teleop import NoCamera, follow


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calibration", default="calib/pose_calibration.json")
    ap.add_argument("--port", required=True, help="the Pico's serial port, e.g. /dev/ttyACM0 or COM19")
    ap.add_argument("--listener", default=None, help="default: Rig-controller's packaged one")
    ap.add_argument("--rig-code", default=None, help="only if Rig-controller isn't pip-installed")
    ap.add_argument("--camera", type=int, default=-1, help="TacTip camera index (-1: none)")
    ap.add_argument("--texture", default="flat", help="simulated block: flat, 1..20 or an STL")
    ap.add_argument("--margin", type=float, default=20.0)
    args = ap.parse_args()

    # the map from simulated pose (mm) to stepper position (steps from home)
    pose_map = PoseMap.load(args.calibration)
    print(pose_map.summary())

    # the simulated rig, over the same block as the real one
    texture = flat_block_stl() if args.texture == "flat" else \
        (int(args.texture) if args.texture.isdigit() else args.texture)
    sim = SimRig(textures=[texture])

    # the real rig: Rig-controller's MicroPython board, and the TacTip camera if there is one
    controller = connect_controller(args.port, args.listener, args.rig_code)
    camera = open_camera(args.camera) if args.camera >= 0 else NoCamera()
    real = RealRig(controller, camera, steps_per_mm=pose_map.steps_per_mm()[0], track=False,
                   settle_s=0.0)

    try:
        follow(sim, real, pose_map, margin_mm=args.margin)
    finally:
        sim.close()
        real.close()


if __name__ == "__main__":
    main()
