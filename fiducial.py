import numpy as np
from nptyping import Array

class Fiducial(object):
    '''
    This class defines a generic fiducial which is identified by points and its
    numerical id. To initialize the fiducial a nump array of points is given,
    such that the points are the columns, i.e.
    [[x1, x2, x3, ...], [y1, y2, y3, ...], [z1, z2, z3, ...]]
    '''
    def __init__(self, pose: Array[np.float64, 3, ...], id: int) -> None:
        self._pose = pose
        self._id = id

    @property
    def pose(self) ->  Array[np.float64, 3, ...]:
        return self._pose

    @property
    def id(self) -> int:
        return self._id

    def __str__(self):
        return "id: {self._id}\tpose={self._pose.T}".format(self=self)
