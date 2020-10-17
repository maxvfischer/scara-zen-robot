from typing import List
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from typing import Callable
from matplotlib.animation import FuncAnimation


class ScaraRobot:
    def __init__(self,
                 length_first_arm: int,
                 length_second_arm: int,
                 angle_per_motor_step: float,
                 art_lambda_function: Callable,
                 timeout_between_steps_sec: float = 1,
                 number_of_steps: int = 1000,
                 visualization_mode: bool = False):
        self.length_first_arm = length_first_arm
        self.length_second_arm = length_second_arm
        self.angle_per_motor_step = angle_per_motor_step

        self.arm_route_x_y_positions = self._generate_x_y_art(
            lambda_function=art_lambda_function,
            num_steps=number_of_steps
        )
        self.timeout_between_steps_sec = timeout_between_steps_sec

        self.current_route_index = 0
        self.previous_actual_angle_first_arm = 0.
        self.previous_actual_angle_second_arm = 0.

        if visualization_mode:
            plt.style.use("ggplot")
            absolute_value_axis_limits = self.length_first_arm + self.length_second_arm
            self.fig = plt.figure()
            self.ax = plt.axes(
                xlim=(-absolute_value_axis_limits, absolute_value_axis_limits),
                ylim=(-absolute_value_axis_limits, absolute_value_axis_limits)
            )

            x_y_first_arm, x_y_second_arm = self._compute_x_y_position_arms()
            self.x_y_trajectory = deque([(x_y_second_arm['x'][1], x_y_second_arm['y'][1])])
            self.x_trajectory = deque([x_y_second_arm['x'][1]], maxlen=number_of_steps)
            self.y_trajectory = deque([x_y_second_arm['y'][1]], maxlen=number_of_steps)

            self.first_arm_plot, = self.ax.plot(x_y_first_arm['x'], x_y_first_arm['y'])
            self.second_arm_plot, = self.ax.plot(x_y_second_arm['x'], x_y_second_arm['y'])
            self.x_y_trajectory_plot = self.ax.scatter(x=x_y_second_arm['x'][1], y=x_y_second_arm['y'][1])

    @staticmethod
    def _generate_x_y_art(lambda_function: Callable,
                          num_steps: int,
                          lower_limit: float = 0,
                          upper_limit: float = 2*np.pi):
        """
        Generates a list of (x, y) coordinates, generated from a polar equation implemented as a lambda function.

        Parameters
        ----------
        lambda_function : Callable
            Radius polar equation implemented as a lambda function. The lambda function is only allowed to include
            the parameter "theta", and shall return a single float or integer.

            Example: lambda_function = eval("lambda theta: 3 + np.sin(10*theta)")

        num_steps : int
            Number of samples to generate between lower_limit and upper_limit

        lower_limit : float
            Starting value in the list of theta

        upper_limit : float
            Ending value in the list of theta

        Return
        ------
            list
                List of tuples, containing (x, y) coordinates
        """
        import inspect
        lambda_args = inspect.getfullargspec(lambda_function).args
        assert len(lambda_args) == 1 and lambda_args[0] == 'theta', \
            "lambda_function is only allowed to have one parameter: 'theta'"

        theta_list = np.linspace(lower_limit, upper_limit, num_steps)
        coordinates = []
        for theta in theta_list:
            r = lambda_function(theta=theta)
            coordinates.append((r * np.cos(theta), r * np.sin(theta)))
        return coordinates

    def _current_numeric_angle_first_arm(self, degrees=True):
        """
        Computes the current numeric angle of the first arm, using inverse kinematics.

        Parameters
        ----------
        degrees : bool
            If the angle shall be returned as radians or degrees

        Return
        ------
        float
            Current numeric angle of the first arm
        """
        x = self.arm_route_x_y_positions[self.current_route_index][0]
        y = self.arm_route_x_y_positions[self.current_route_index][1]
        rad_angle_second_arm = self._current_numeric_angle_second_arm(degrees=False)

        angle_part_1 = np.arctan2(y, x)
        angle_part_2 = np.arctan2(
            self.length_second_arm * np.sin(rad_angle_second_arm),
            self.length_first_arm + self.length_second_arm*np.cos(rad_angle_second_arm)
        )

        angle = angle_part_1 - angle_part_2

        if degrees:
            angle = angle*(180/np.pi)
            angle = angle % 360

        return angle

    def _current_numeric_angle_second_arm(self, degrees=True):
        """
        Computes the current numeric angle of the second arm, using inverse kinematics.

        Parameters
        ----------
        degrees : bool
            If the angle shall be returned as radians or degrees

        Return
        ------
        float
            Current numeric angle of the second arm
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

    def _num_step_first_motor(self):
        current_numeric_angle_first_arm = self._current_numeric_angle_first_arm()
        diff_numeric_actual_first_arm = current_numeric_angle_first_arm - self.previous_actual_angle_first_arm
        print(f"Arm 1 - Diff: {diff_numeric_actual_first_arm} | Curr: {current_numeric_angle_first_arm} | Prev: {self.previous_actual_angle_first_arm}")
        direction = 1 if diff_numeric_actual_first_arm >= 0 else -1
        num_steps = np.abs(diff_numeric_actual_first_arm//self.angle_per_motor_step)

        return direction, num_steps

    def _num_step_second_motor(self):
        current_numeric_angle_second_arm = self._current_numeric_angle_second_arm()
        diff_numeric_actual_second_arm = current_numeric_angle_second_arm - self.previous_actual_angle_second_arm
        print(f"Arm 2 - Diff: {diff_numeric_actual_second_arm} | Curr: {current_numeric_angle_second_arm} | Prev: {self.previous_actual_angle_second_arm}")
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
        self.x_y_trajectory_plot.set_offsets(list(self.x_y_trajectory))

        return self.first_arm_plot, self.second_arm_plot, self.x_y_trajectory_plot

    def _visualization_step(self, i):
        self._step()

        x = self.arm_route_x_y_positions[self.current_route_index][0]
        y = self.arm_route_x_y_positions[self.current_route_index][1]
        print(f"x: {x} | y: {y}")
        x_y_first_arm, x_y_second_arm = self._compute_x_y_position_arms()
        self.x_trajectory.append(x_y_second_arm['x'][1])
        self.y_trajectory.append(x_y_second_arm['y'][1])
        self.x_y_trajectory.append((x_y_second_arm['x'][1], x_y_second_arm['y'][1]))

        self.first_arm_plot.set_data(x_y_first_arm['x'], x_y_first_arm['y'])
        self.second_arm_plot.set_data(x_y_second_arm['x'], x_y_second_arm['y'])
        self.x_y_trajectory_plot.set_offsets(list(self.x_y_trajectory))

    def _step(self):
        x = self.arm_route_x_y_positions[self.current_route_index][0]
        y = self.arm_route_x_y_positions[self.current_route_index][1]

        dir_first_arm, steps_first_arm = self._num_step_first_motor()
        dir_second_arm, steps_second_arm = self._num_step_second_motor()

        self.previous_actual_angle_first_arm = (self.previous_actual_angle_first_arm +
                                                dir_first_arm * steps_first_arm * self.angle_per_motor_step) % 360

        self.previous_actual_angle_second_arm = (self.previous_actual_angle_second_arm +
                                                 dir_second_arm * steps_second_arm * self.angle_per_motor_step) % 360

        #print(f"x_in_list: {x} | y_in_list: {y}")
        #print(f"Dir arm 1: {dir_first_arm} | Dir arm 2: {dir_second_arm}")
        #print(f"Steps arm 1: {steps_first_arm} | Steps arm 2: {steps_second_arm}")
        #print(f"Angle arm 1: {self.previous_actual_angle_first_arm} | Angle arm 2: {self.previous_actual_angle_second_arm}")
        print()
        if self.current_route_index < len(self.arm_route_x_y_positions) - 1:
            self.current_route_index += 1
        else:
            self.current_route_index = 0

    def start(self):
        animation_interval = len(self.arm_route_x_y_positions) - 1
        anim = FuncAnimation(self.fig, self._visualization_step, init_func=self._init_function_visualization,
                             interval=200, blit=False)
        #anim.save('line.gif', dpi=80, writer='imagemagick')
        plt.show()

    def stop(self):
        raise NotImplementedError
