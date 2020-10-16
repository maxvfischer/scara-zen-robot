from typing import List
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class ScaraRobot:
    def __init__(self,
                 length_first_arm: int,
                 length_second_arm: int,
                 angle_per_motor_step: float,
                 arm_route_x_y_positions: List[tuple],
                 timeout_between_steps_sec: float = 1,
                 visualization_mode: bool = False):
        self.length_first_arm = length_first_arm
        self.length_second_arm = length_second_arm
        self.angle_per_motor_step = angle_per_motor_step

        self.arm_route_x_y_positions = arm_route_x_y_positions
        self.timeout_between_steps_sec = timeout_between_steps_sec
        self.visualization_mode = visualization_mode

        self.current_route_index = 0
        self.previous_actual_angle_first_arm = 0.
        self.previous_actual_angle_second_arm = 0.

        if visualization_mode:
            plt.style.use("ggplot")
            absolute_value_axis_limits = 7
            self.fig = plt.figure()
            self.ax = plt.axes(
                xlim=(-absolute_value_axis_limits, absolute_value_axis_limits),
                ylim=(-absolute_value_axis_limits, absolute_value_axis_limits)
            )

            x_y_first_arm, x_y_second_arm = self._compute_x_y_position_arms()

            self.first_arm_plot, = self.ax.plot(x_y_first_arm['x'], x_y_first_arm['y'])
            self.second_arm_plot, = self.ax.plot(x_y_second_arm['x'], x_y_second_arm['y'])


    @staticmethod
    def _origo_x_y_distance(x_coordinate: float, y_coordinate: float):
        return np.sqrt(x_coordinate**2 + y_coordinate**2)

    def _current_numeric_angle_first_arm(self, degrees=True):
        x = self.arm_route_x_y_positions[self.current_route_index][0]
        y = self.arm_route_x_y_positions[self.current_route_index][1]
        second_angle = self._current_numeric_angle_second_arm(degrees=False)

        angle_part_1 = np.arctan2(y, x)
        angle_part_2 = np.arctan2(
            self.length_second_arm * np.sin(second_angle),
            self.length_first_arm + self.length_second_arm*np.cos(second_angle)
        )

        angle = angle_part_1 - angle_part_2

        if degrees:
            angle = angle*(180/np.pi)
            angle = angle % 360

        return angle

    def _current_numeric_angle_second_arm(self, degrees=True):
        """
        Compute the inverse kinematics of the second Scara robot arm.
        """
        x = self.arm_route_x_y_positions[self.current_route_index][0]
        y = self.arm_route_x_y_positions[self.current_route_index][1]

        cos_angle = (x**2 + y**2 - self.length_first_arm**2 - self.length_second_arm**2) / \
                    (2*self.length_first_arm*self.length_second_arm)
        angle = np.arccos(cos_angle)

        if degrees:
            angle = angle*(180/np.pi)
            angle = angle % 360

        return angle

    def _num_step_first_motor(self, current_numeric_angle_first_arm: float):
        diff_numeric_actual_first_arm = current_numeric_angle_first_arm - self.previous_actual_angle_first_arm
        direction = 1 if diff_numeric_actual_first_arm >= 0 else -1
        num_steps = np.abs(diff_numeric_actual_first_arm//self.angle_per_motor_step)

        return direction, num_steps

    def _num_step_second_motor(self, current_numeric_angle_second_arm: float):
        diff_numeric_actual_second_arm = current_numeric_angle_second_arm - self.previous_actual_angle_second_arm
        direction = 1 if diff_numeric_actual_second_arm >= 0 else -1
        num_steps = np.abs(diff_numeric_actual_second_arm // self.angle_per_motor_step)

        return direction, num_steps

    def _compute_x_y_position_arms(self):
        x_y_first_arm = {
            'x': [0, self.length_first_arm * np.cos(self.previous_actual_angle_first_arm*(np.pi/180))],
            'y': [0, self.length_first_arm * np.sin(self.previous_actual_angle_first_arm*(np.pi/180))]
        }

        x_y_second_arm = {
            'x': [
                x_y_first_arm['x'][1],
                x_y_first_arm['x'][1] + self.length_second_arm * np.cos(
                    (self.previous_actual_angle_first_arm+self.previous_actual_angle_second_arm)*(np.pi/180))
            ],
            'y': [
                x_y_first_arm['y'][1],
                x_y_first_arm['y'][1] + self.length_second_arm * np.sin(
                    (self.previous_actual_angle_first_arm+self.previous_actual_angle_second_arm)*(np.pi/180))]
        }

        return x_y_first_arm, x_y_second_arm

    def _init_function_visualization(self):
        x_y_first_arm, x_y_second_arm = self._compute_x_y_position_arms()
        self.first_arm_plot.set_data(x_y_first_arm['x'], x_y_first_arm['y'])
        self.second_arm_plot.set_data(x_y_second_arm['x'], x_y_second_arm['y'])

        return self.first_arm_plot, self.second_arm_plot

    def _visualization_step(self, i):
        self._step()

        x_y_first_arm, x_y_second_arm = self._compute_x_y_position_arms()
        self.first_arm_plot.set_data(x_y_first_arm['x'], x_y_first_arm['y'])
        self.second_arm_plot.set_data(x_y_second_arm['x'], x_y_second_arm['y'])

        time.sleep(self.timeout_between_steps_sec)

    def _step(self):
        x = self.arm_route_x_y_positions[self.current_route_index][0]
        y = self.arm_route_x_y_positions[self.current_route_index][1]

        numeric_angle_first_arm = self._current_numeric_angle_first_arm()
        numeric_angle_second_arm = self._current_numeric_angle_second_arm()

        dir_first_arm, steps_first_arm = self._num_step_first_motor(
            current_numeric_angle_first_arm=numeric_angle_first_arm
        )

        dir_second_arm, steps_second_arm = self._num_step_second_motor(
            current_numeric_angle_second_arm=numeric_angle_second_arm
        )

        self.previous_actual_angle_first_arm = (self.previous_actual_angle_first_arm +
                                                dir_first_arm * steps_first_arm * self.angle_per_motor_step) % 360

        self.previous_actual_angle_second_arm = (self.previous_actual_angle_second_arm +
                                                 dir_second_arm * steps_second_arm * self.angle_per_motor_step) % 360

        print(f"First arm: {dir_first_arm} | {steps_first_arm}")
        print(f"Second arm: {dir_second_arm} | {steps_second_arm}")
        print(f"First num angle: {numeric_angle_first_arm}")
        print(f"Second num angle: {numeric_angle_second_arm}")
        print(f"Prev angle first arm: {self.previous_actual_angle_first_arm}")
        print(f"Prev angle second arm: {self.previous_actual_angle_second_arm}")
        print()

        if self.current_route_index < len(self.arm_route_x_y_positions) - 1:
            self.current_route_index += 1
        else:
            self.current_route_index = 0

    def start(self):
        #self._step()
        #time.sleep(self.timeout_between_steps_sec)
        animation_interval = len(self.arm_route_x_y_positions) - 1
        anim = FuncAnimation(self.fig, self._visualization_step, init_func=self._init_function_visualization,
                             frames=200, interval=animation_interval, blit=False)
        plt.show()

    def stop(self):
        raise NotImplementedError

def generate_infinite_sign(n=100):
    import numpy as np
    theta_list = np.linspace(0, 2*np.pi)
    coordinates = []
    for theta in theta_list:
        r = 3 + np.sin(10*theta)
        coordinates.append((r*np.cos(theta), r*np.sin(theta)))
    return coordinates