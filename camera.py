import sys
from typing import Optional, List
import numpy as np
from nptyping import Array
from math import pi, atan, sqrt, isclose
import yaml

class Camera(object):
    '''
    Projective camera model with
        - camera intrinsic and extrinsic parameters handling
        - projection of camera coordinates to an image
        - conversion of image coordinates on a plane to camera coordinates
        - visibility handling
    '''
    def __init__(self) -> None:
        self._camera_matrix = np.eye(4)  # camera intrinsic parameters
        self._R = np.eye(3)
        self._t = np.zeros((3, 1))
        self._aspect_ratio = np.zeros((2,), dtype=int)

    def load(self, filename: str) -> None:
        '''
        Load camera model from a YAML file.
        Example::
            cameraMatrix:
                - [fx, s, cx]
                - [0.0, fy, cy]
                - [0.0, 0.0, 1.0]
            R:
                - [-0.9316877145365, -0.3608289515885, 0.002545329627547]
                - [-0.1725273110187, 0.4247524018287, -0.8888909933995]
                - [0.3296724908378, -0.8263880720441, -0.4579894432589]
            t:
                - [-1.365061486465]
                - [3.431608806127]
                - [17.74182159488]
            aspect_ratio: [960, 768]
        '''
        data = yaml.load(open(filename))
        if 'cameraMatrix' in data and 'R' in data and 't' in data and 'aspect_ratio' in data:
            self._camera_matrix = np.array(data['cameraMatrix'], dtype=np.float64).reshape((3, 3))
            self._R = np.array(data['R'], dtype=np.float64).reshape((3, 3))
            self._t = np.array(data['t'], dtype=np.float64).reshape((3, 1))
            self._aspect_ratio = np.array(data['aspect_ratio']).reshape((2,))
        else:
            error("Nothing loaded from {}, check the contents.".format(filename))
            sys.exit(1)

    def _is_visible_xy(self, image_pts: Array[np.float64, 2, ...]) -> bool:
        '''
        Check the visibility of the image point(s)
        '''
        return (image_pts[0, :] >=0) & (image_pts[1, :] >= 0) & \
               (image_pts[0, :] < self._aspect_ratio[0]) & \
               (image_pts[1, :] < self._aspect_ratio[1])

    def is_visible(self, R_robot: Array[np.float64, 3, 3],
                   t_robot: Array[np.float64, 3, 1],
                   world_pts: Array[np.float64, 3, ...]) -> bool:
        '''
        Check the visibility of the world point(s)
        '''
        world_pts_h = np.append(world_pts, [np.ones_like(world_pts[0, :])], axis=0)  # [4x1]
        world_to_robot = np.append(np.append(R_robot, t_robot, axis=1), [[0, 0, 0, 1]], axis=0) # [4x4]
        robot_pts_h = world_to_robot.dot(world_pts_h)  # [4x1]

        # this matrix swaps the x and z axes, and reverses the y axis
        axes_swap = np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
        robot_pts_h_swapped = axes_swap.dot(robot_pts_h)  # [4x1]

        robot_to_camera = np.append(self._R, self._t, axis=1) # [3x4]
        camera_pts = robot_to_camera.dot(robot_pts_h_swapped)  # [3x1]


        if not all(camera_pts[2, :] > 1e-15):
            return False

        image_pts = self._camera_matrix.dot(camera_pts)  # [[u], [v], [w]]
        image_pts_h = ((image_pts.T / image_pts[-1, :].reshape((image_pts.shape[1],1))).T)  # [[u/w], [v/w]]

        return all(camera_pts[2, :] > 1e-15) and all(self._is_visible_xy(image_pts_h))
