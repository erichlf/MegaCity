from typing import Optional, List, Set
import numpy as np
from nptyping import Array
from math import sin, cos, pi
from camera import Camera
from fiducial import Fiducial
import random

class Robot(object):
    def __init__(self, id: int,
                 pose: Optional[Array[np.float64, 1, 3]]=np.zeros((3,)),
                 sd: Optional[float]=0.1,
                 world_size: Optional[Array[int, 1, 2]]=np.array([800, 800])) -> None:
        self._id = id
        self._R = np.ones((3, 3))
        self._t = np.zeros((3, 1))
        self._pose = pose
        self._cameras = []  # collection of cameras
        self._sd = sd  # standard deviation for noisy movement
        self._world_size = world_size  # 2D limits of the world in meters
        # get the transform from robot to world coordinates
        self._update_robotMatrix()

    def _update_robotMatrix(self) -> None:
        '''
        Takes the current pose and creates our robot matrix from it
        '''
        self._R = np.array([[cos(self._pose[2]), -sin(self._pose[2]), 0],
                            [sin(self._pose[2]), cos(self._pose[2]), 0],
                            [0, 0, 1]],
                           dtype=np.float64)
        self._t = np.array([[self._pose[0]],
                            [self._pose[1]],
                            [0]], dtype=np.float64)

    def set_pose(self, pose: Array[np.float64, 3, 1]) -> None:
        '''
        Set the pose of the robot (x, y, theta)
        '''
        self._pose = pose
        self._pose[2] %= (2 * pi)
        self._pose[0] %= self._world_size[0]
        self._pose[1] %= self._world_size[1]

        self._update_robotMatrix()

    def id(self) -> int:
        '''
        Gets the id of this robot
        '''
        return self._id

    def pose(self) -> Array[np.float64, 3, 1]:
        '''
        Gets the x, y coordinate and the direction the robot is facing
        '''
        return self._pose

    def attach_camera(self, camera: Camera) -> None:
        '''
        Attaches a camera object. This object should have the robot to camera
        transform already defined within it object.
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

        self._update_robotMatrix()

    def find_fiducials(self, fiducials: List[Fiducial]) -> Set[Fiducial]:
        '''
        Find the fiducials which are in the camera view
        '''
        seen = set()
        for fiducial in fiducials:
            for camera in self._cameras:
                if camera.is_visible(self._R, self._t, fiducial.pose()):
                    seen = seen.union([fiducial])  # add to seen list
                    break  # don't need to find the fiducial more than once

        return seen
