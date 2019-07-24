from typing import Optional, List, Tuple
import numpy as np
from nptyping import Array
from math import sin, cos, pi
from camera import Camera
from fiducial import Fiducial
import random

class Robot(object):
    def __init__(self, pose: Optional[Array[np.float64, 1, 3]]=np.zeros((3,)),
                 sd: Optional[float]=0.1,
                 world_size: Optional[Array[int, 1, 2]]=np.array([800, 800])) -> None:
        self._pose = pose
        self._cameras = []  # collection of cameras
        self._sd = sd  # standard deviation for noisy movement
        self._world_size = world_size  # 2D limits of the world in meters

    def set_pose(self, pose: Array[np.float64, 3, 1]) -> None:
        '''
        Set the pose of the robot (x, y, theta)
        '''
        self._pose = pose
        self._pose[2] %= (2 * pi)

    def pose(self) -> Array[np.float64, 3, 1]:
        '''
        Gets the x, y coordinate and the direction the robot is facing
        '''
        return self._pose

    def attach_camera(self, camera: Camera) -> None:
        '''
        Attaches a camera facing the relative diection, angle, to the robot
        '''
        self._cameras.append(camera)

    def move(self, turn: float, forward: float) -> None:
        '''
        Given the angle (turn) and the distance to travel (forward) move the
        robot that direction and distance
        '''
        forward += random.uniform(0.0, self._sd)  # add some noise to our forward movement
        turn += random.uniform(0.0, self._sd)  # add some noise to our turn
        self._pose[2] += turn
        self._pose[0] += forward * cos(self._pose[2])
        self._pose[1] += forward * sin(self._pose[2])

        self._pose[2] %= (2 * pi)
        self._pose[0] %= self._world_size[0]
        self._pose[1] %= self._world_size[1]

    def find_fiducials(self, fiducials: List[Fiducial]) -> List[Fiducial]:
        '''
        Find the fiducials which are in the camera view
        '''
        seen = []
        for fiducial in fiducials:
            for camera in self._cameras:
                if camera.is_visible(fiducial.pose()):
                    seen.append(fiducial.pose())  # add to seen list
                    break  # don't need to find the fiducial more than once

        return seen
