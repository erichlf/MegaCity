import sys
import json
import random
import argparse
import numpy as np
from math import sqrt, pi

from robot import Robot
from camera import Camera
from fiducial import Fiducial
from simulator import Simulator


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--unit-config', type=str, required=True)
    parser.add_argument('-a', '--app-config', type=str, required=True)
    args = parser.parse_args()

    with open(args.app_config, 'r') as f:
        sim_config = json.load(f)["simulation"]
    with open(args.unit_config, 'r') as f:
        unit_config = json.load(f)

    robots = list()
    robots.append(Robot(0, pose=np.array([[0], [0], [0]], dtype=np.float64),
                        world_size=np.array([sim_config["world"]["length_m"], sim_config["world"]["width_m"]])))
    robots.append(Robot(1, pose=np.array([[0], [0], [pi / 4]], dtype=np.float64),
                        world_size=np.array([sim_config["world"]["length_m"], sim_config["world"]["width_m"]])))
    robots.append(Robot(2, pose=np.array([[0], [0], [pi / 2]], dtype=np.float64),
                        world_size=np.array([sim_config["world"]["length_m"], sim_config["world"]["width_m"]])))
    robots.append(Robot(3, pose=np.array([[0], [0], [3 * pi / 4]], dtype=np.float64),
                        world_size=np.array([sim_config["world"]["length_m"], sim_config["world"]["width_m"]])))
    robots.append(Robot(4, pose=np.array([[0], [0], [pi]], dtype=np.float64),
                        world_size=np.array([sim_config["world"]["length_m"], sim_config["world"]["width_m"]])))
    robots.append(Robot(5, pose=np.array([[0], [0], [5 * pi / 4]], dtype=np.float64),
                        world_size=np.array([sim_config["world"]["length_m"], sim_config["world"]["width_m"]])))
    robots.append(Robot(6, pose=np.array([[0], [0], [3 * pi / 2]], dtype=np.float64),
                        world_size=np.array([sim_config["world"]["length_m"], sim_config["world"]["width_m"]])))
    robots.append(Robot(7, pose=np.array([[0], [0], [7 * pi / 4]], dtype=np.float64),
                        world_size=np.array([sim_config["world"]["length_m"], sim_config["world"]["width_m"]])))
    robots.append(Robot(8, pose=np.array([[-1], [1], [0]], dtype=np.float64),
                        world_size=np.array([sim_config["world"]["length_m"], sim_config["world"]["width_m"]])))
    robots.append(Robot(9, pose=np.array([[-1], [1], [pi / 4]], dtype=np.float64),
                        world_size=np.array([sim_config["world"]["length_m"], sim_config["world"]["width_m"]])))

    for robot in robots:
        # create, initialize, and attach camera to robot
        camera = Camera(intrinsics=unit_config["camera"]["0"]["intrinsics"],
                        extrinsics=unit_config["camera"]["0"]["extrinsics"])
        robot.attach_camera(camera)

    # generate 10 fiducials and place them
    fiducials = []
    '''
    for i in range(0,10):
        fiducial = Fiducial(np.array([[random.randrange(0, world_length),
                                       random.randrange(0, world_width),
                                       random.randrange(0, world_height)]],
                                     dtype=np.float64), i)
        fiducials.append(fiducial)
    '''
    fiducials.append(Fiducial(np.array([[1, 1, 1, 1],
                                        [0, 0.1, 0.1, 0],
                                        [1, 1, 0.9, 0.9]], dtype=np.float64), 0))
    fiducials.append(Fiducial(np.array([[1], [1], [1]], dtype=np.float64), 1))
    fiducials.append(Fiducial(np.array([[0], [1], [1]], dtype=np.float64), 2))
    fiducials.append(Fiducial(np.array([[-1], [1], [1]], dtype=np.float64), 3))
    fiducials.append(Fiducial(np.array([[-1], [0], [1]], dtype=np.float64), 4))
    fiducials.append(Fiducial(np.array([[-1], [-1], [1]], dtype=np.float64), 5))
    fiducials.append(Fiducial(np.array([[0], [-1], [1]], dtype=np.float64), 6))
    fiducials.append(Fiducial(np.array([[1], [-1], [1]], dtype=np.float64), 7))

    seen = Simulator(robots, fiducials, world_size=np.array([sim_config["world"]["length_m"],
                                                             sim_config["world"]["width_m"]])).run()  # run the simulation


if __name__ == '__main__':
    sys.exit(main())
