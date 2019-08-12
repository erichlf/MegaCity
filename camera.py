import sys
import yaml
import numpy as np
from nptyping import Array
from typing import Optional, List
from math import pi, atan, sqrt, isclose


class Camera(object):
    '''
    Projective camera model with
        - camera intrinsic and extrinsic parameters handling
        - projection of camera coordinates to an image
        - conversion of image coordinates on a plane to camera coordinates
        - visibility handling
    '''
    def __init__(self, intrinsics, extrinsics=None) -> None:
        self._camera_matrix = np.array([
            [intrinsics["fx"], 0, intrinsics["cx"]],
            [0, intrinsics["fy"], intrinsics["cy"]],
            [0, 0, 1]])

        self._aspect_ratio = np.array([
            intrinsics["image_size"]["height"],
            intrinsics["image_size"]["width"]])

        if extrinsics is None:
            self._R = np.eye(3)
            self._t = np.zeros((3, 1))
        else:
            # TODO: Convert axis angle rotation to 3x3 matrix
            self._R = np.eye(3)
            self._t = np.array([
                [extrinsics["x"]],
                [extrinsics["y"]],
                [extrinsics["z"]]])

    def _is_visible_xy(self, image_pts: Array[np.float64, 2, ...]) -> bool:
        '''
        Check the visibility of the image point(s)
        '''
        return (image_pts[0, :] >= 0) & (image_pts[1, :] >= 0) & \
               (image_pts[0, :] < self._aspect_ratio[0]) & \
               (image_pts[1, :] < self._aspect_ratio[1])

    def is_visible(self, R_robot: Array[np.float64, 3, 3],
                   t_robot: Array[np.float64, 3, 1],
                   world_pts: Array[np.float64, 3, ...]) -> bool:
        '''
        Check the visibility of the world point(s)
        '''
        world_pts_h = np.append(world_pts, [np.ones_like(world_pts[0, :])], axis=0)

        robot_to_world = np.append(np.append(R_robot, t_robot, axis=1), [[0, 0, 0, 1]], axis=0)
        camera_to_robot = np.append(np.append(self._R, self._t, axis=1), [[0, 0, 0, 1]], axis=0)
        camera_to_world = camera_to_robot.dot(robot_to_world)
        # swaps axes to correspond to the image frame
        axes_swap = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
        swapped_camera_to_world = camera_to_world.dot(axes_swap)
        world_to_camera_swapped = np.linalg.inv(swapped_camera_to_world)
        camera_pts = world_to_camera_swapped.dot(world_pts_h)

        if not all(camera_pts[2, :] > 1e-15):
            return False

        image_pts = self._camera_matrix.dot(camera_pts[0:3, :])  # [[u], [v], [w]]
        image_pts_h = ((image_pts.T / image_pts[-1, :].reshape((image_pts.shape[1],1))).T)  # [[u/w], [v/w]]

        return all(camera_pts[2, :] > 1e-15) and all(self._is_visible_xy(image_pts_h))
