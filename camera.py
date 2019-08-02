import sys
from typing import Optional, List
import numpy as np
from nptyping import Array
from math import pi, atan, sqrt
import yaml
import cv2

class Camera(object):
    '''
    Projective camera model with
        - camera intrinsic and extrinsic parameters handling
        - projection of camera coordinates to an image
        - conversion of image coordinates on a plane to camera coordinates
        - visibility handling
    '''
    def __init__(self) -> None:
        self._cameraMatrix = np.eye(3)  # camera intrinsic parameters
        self._R = np.eye(3)
        self._t = np.zeros((3, 1))
        self._aspect_ratio = np.zeros((2,), dtype=int)

        self._update_camera_matrix()

    def _update_camera_matrix(self) -> None:
        '''
        Update camera_matrix from K, R and t.
        '''
        self._camera_matrix = self._cameraMatrix.dot(np.hstack((self._R, self._t)))

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
            aspect_ratio: [1920, 1080]
        '''
        data = yaml.load(open(filename))
        if 'cameraMatrix' in data and 'R' in data and 't' in data and 'aspect_ratio' in data:
            self._cameraMatrix = np.array(data['cameraMatrix'], dtype=np.float64).reshape((3, 3))
            self._R = np.array(data['R'], dtype=np.float64).reshape((3, 3))
            self._t = np.array(data['t'], dtype=np.float64).reshape((3, 1))
            self._aspect_ratio = np.array(data['aspect_ratio']).reshape((2,))
        else:
            error('Nothing loaded from %s, check the contents.' % filename)
            sys.exit(1)
        self._update_camera_matrix()

    def get_principle_point(self) -> Array[np.float64, 1, 2]:
        '''
        Get the principle point from our camera
        '''
        return self._cameraMatrix[0:2, 2].reshape((1, 2))

    def _is_visible_xy(self, image_pts: Array[np.float64, ..., 2]) -> bool:
        '''
        Check the visibility of the image point(s)
        '''
        # @TODO: make sure object is not behind camera
        return (image_pts[:, 0] >=0) & (image_pts[:, 1] >= 0) & \
               (image_pts[:, 0] < self._aspect_ratio[0]) & \
               (image_pts[:, 1] < self._aspect_ratio[1])

    def is_visible(self, R_robot: Array[np.float64, 3, 3],
                   t_robot: Array[np.float64, 3, 1],
                   world_pts: Array[np.float64]) -> bool:
        '''
        Check the visibility of the world point(s)
        '''
        return self._is_visible_xy(self._world_to_image(R_robot, t_robot, world_pts))

    def _world_to_image(self, R_robot: Array[np.float64, 3, 3],
                        t_robot: Array[np.float64, 3, 1],
                        world_pts: Array[np.float64, ..., 3]) -> Array[np.float64, ..., 2]:
        '''
        Project world coordinates to image coordinates.
        '''
        # The world to camera transform, in block form, is
        # [[R_robot, t_robot], [0, 1]] * [[self._R, self._t], [0, 1]]
        # = [[R_robot * self._R, R_robot * self._t + t_robot], [0, 1]]
        image_coords = cv2.projectPoints(world_pts,
                                         cv2.Rodrigues(np.matmul(R_robot, self._R))[0],
                                         np.matmul(R_robot, self._t) + t_robot,
                                         self._cameraMatrix, None)[0].reshape(-1, 2)

        return image_coords # if homogeneous else homogeneous_to_euclidean(image_coords)

def homogeneous_to_euclidean(homogeneous: Array[np.float64, ..., 4]) -> Array[np.float64]:
    '''
    Convert homogeneous coordinates to euclidean coordinates
    '''
    return (homogeneous / homogeneous[:, -1])[:, 0:-1]

def euclidean_to_homogeneous(euclidean: Array[np.float64, ..., 3]) -> Array[np.float64, ..., 4]:
    '''
    Convert euclidean coordinates to homogeneous coordinates
    '''
    return np.append(euclidean, np.ones((np.size(euclidean, 0), 1)), axis=1)
