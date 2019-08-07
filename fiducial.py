import numpy as np
from nptyping import Array

class Fiducial(object):
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
