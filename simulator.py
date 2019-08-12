from typing import List, Set, Optional
import numpy as np
from nptyping import Array
from robot import Robot
from fiducial import Fiducial
from scipy.spatial import KDTree
import random
from math import pi, cos, sin

class Simulator(object):
    def __init__(self, robots: List[Robot], fiducials: List[Fiducial],
                 world_size: Optional[Array[int, 1, 2]]=np.array([800, 800])) -> None:
        self._monte_carlo_steps = 1000
        self._robots = robots
        self._fiducials = fiducials
        self._state = {robot: robot.pose for robot in self._robots}
        self._world_size = world_size

    def simulate_true_move(self, robot: Robot, turn: float, forward: float) -> None:
        '''
        Given the angle (turn) and the distance to travel (forward) move the
        robot that direction and distance
        This function allows us to keep track of the true position of the robots
        '''
        x = self._state[robot][0]
        y = self._state[robot][1]
        theta = self._state[robot][2]

        theta = theta + turn
        x = x + forward * cos(theta)
        y = y - forward * sin(theta)

        theta %= (2 * pi)
        x %= self._world_size[0]
        y %= self._world_size[1]

        self._state[robot] = (x, y, theta)

    def run(self) -> List[Fiducial]:
        seen = []
        for robot in self._robots:
            # for step in range(1, self._monte_carlo_steps + 1):
                # get a random turn and forward
                # delta = (random.uniform(0, 2 * pi), random.uniform(0, 1))
                # have the robot move to the true location
                # self.simulate_true_move(robot, delta[0], delta[1])
                # have the robot itself move (robot will move with some error)
                # robot.move(delta[0], delta[1])
            found = robot.find_fiducials(self._fiducials)
            seen.append(found)

        return seen
