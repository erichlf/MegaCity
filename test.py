import sys
from simulator import Simulator
from robot import Robot
from camera import Camera
from fiducial import Fiducial
import numpy as np
import random

def main() -> int:
    world_width  = 800  # in meters
    world_length = 800  # in meters
    world_height  = 3  # in meters
    robot = Robot(world_size=np.array([world_length, world_width]))
    camera = Camera()
    camera.load('camera.yaml')  # this will set our camera matrix
    robot.attach_camera(camera)
    # generate 10 fiducials and place them
    fiducials = []
    for i in range(0,10):
        fiducial = Fiducial(np.array([[random.randrange(0, world_length),
                                       random.randrange(0, world_width),
                                       random.randrange(0, world_height)]],
                                     dtype=np.float64), i)
        fiducials.append(fiducial)

    Simulator([robot], fiducials, world_size=np.array([world_length,
                                                       world_width])).run()  # run the simulation

    return 0

if __name__ == '__main__':
    sys.exit(main())
