import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from typing import Callable, List, Tuple
from matplotlib.animation import FuncAnimation


class ScaraRobot:
    """
    SCARA robot using two arms.

    See the paper "Formal Kinematic Analysis of the Two-Link Planar Manipulator"
    (https://www.researchgate.net/publication/283377443_Formal_Kinematic_Analysis_of_the_Two-Link_Planar_Manipulator)
    for inverse kinematic equations used.

    Arguments
    ---------
    length_first_arm : int
        Length of first arm.

    length_second_arm : int
        Length of second arm.

    angle_per_motor_step : float
        Number of degrees per step by the stepper motor.

    art_config : dict
        Config of the polar equation graph that is going to be drawn by the SCARA robot.

    number_of_steps : int
        Number of steps to complete one full 'circle' of the polar equation graph.

    visualization_mode : bool
        Visualization mode.
    """
    def __init__(self,
                 length_first_arm: int,
                 length_second_arm: int,
                 angle_per_motor_step: float,
                 art_config: dict,
                 number_of_steps: int = 1000,
                 visualization_mode: bool = False):
        self.length_first_arm = length_first_arm
        self.length_second_arm = length_second_arm
        self.angle_per_motor_step = angle_per_motor_step

        self.arm_route_x_y_positions = self._generate_x_y_art(
            lambda_function=art_config['func'],
            lower_limit=art_config['lower_limit'],
            upper_limit=art_config['upper_limit'],
            scale=art_config['scale'],
            num_steps=number_of_steps
        )

        self.current_route_index = 0
        self.current_actual_angle_first_arm = 0.
        self.current_actual_angle_second_arm = 0.
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
            self.x_y_trajectory_plot, = self.ax.plot(x_y_second_arm['x'][1], x_y_second_arm['y'][1])

    @staticmethod
    def _generate_x_y_art(lambda_function: Callable,
                          num_steps: int,
                          scale: float = 1,
                          lower_limit: float = 0,
                          upper_limit: float = 2*np.pi) -> List[Tuple[float, float]]:
        """
        Generates a list of (x, y) coordinates, generated from a polar equation implemented as a lambda function.

        Parameters
        ----------
        lambda_function : Callable
            Radius polar equation implemented as a lambda function. The lambda function is only allowed to include
            the parameter "theta", and shall return a single float or integer.

            Example: lambda_function = eval("lambda theta: 3 + np.sin(10*theta)")

        num_steps : int
            Number of samples to generate between lower_limit and upper_limit.

        scale : float
            Scaling factor. Increases size of polar equation graph.

        lower_limit : float
            Starting value in the list of theta.

        upper_limit : float
            Ending value in the list of theta.

        Return
        ------
        list
            List of tuples, containing (x, y) coordinates.
        """
        import inspect
        lambda_args = inspect.getfullargspec(lambda_function).args
        assert len(lambda_args) == 1 and lambda_args[0] == 'theta', \
            "lambda_function is only allowed to have one parameter: 'theta'"

        theta_list = np.linspace(lower_limit, upper_limit, num_steps)
        coordinates = []
        for theta in theta_list:
            r = lambda_function(theta=theta)*scale
            coordinates.append((r * np.cos(theta), r * np.sin(theta)))
        return coordinates

    @staticmethod
    def _difference_two_angles(angle_1: float, angle_2: float) -> float:
        """
        Computes the absolute difference between two angles.

        Parameters
        ----------
        angle_1 : float
            First angle.

        angle_2 : float
            Second angle.

        Return
        ------
        float
            Absolute difference between angle_1 and angle_2.
        """
        diff = np.abs(angle_1 - angle_2)
        diff = 360 - diff if diff > 180 else diff
        return diff

    def _update_current_numeric_angles(self):
        """
        Updates the current numeric angles of the first and second arms.
        """
        positive_angle_first_arm, negative_angle_first_arm = self._current_numeric_angles_first_arm()
        positive_angle_second_arm, negative_angle_second_arm = self._current_numeric_angles_second_arm()

        positive_diff = self._difference_two_angles(self.previous_actual_angle_first_arm, positive_angle_first_arm)
        negative_diff = self._difference_two_angles(self.previous_actual_angle_first_arm, negative_angle_first_arm)

        if positive_diff < negative_diff:
            self.current_actual_angle_first_arm = positive_angle_first_arm
            self.current_actual_angle_second_arm = positive_angle_second_arm
        else:
            self.current_actual_angle_first_arm = negative_angle_first_arm
            self.current_actual_angle_second_arm = negative_angle_second_arm

    def _current_numeric_angles_first_arm(self, degrees=True) -> Tuple[float, float]:
        """
        Computes the two different numeric angles ('Elbow up' and 'Elbow down') of the first arm, using trigonometric
        inverse kinematics.

        Parameters
        ----------
        degrees : bool
            If the angle shall be returned as radians or degrees.

        Return
        ------
        float
            Current numeric angle of the first arm.
        """
        x = self.arm_route_x_y_positions[self.current_route_index][0]
        y = self.arm_route_x_y_positions[self.current_route_index][1]

        positive_rad_angle_second_arm, negative_rad_angle_second_arm = \
            self._current_numeric_angles_second_arm(degrees=False)

        positive_angle_part_1 = np.arctan2(y, x)
        positive_angle_part_2 = np.arctan2(
            self.length_second_arm * np.sin(positive_rad_angle_second_arm),
            self.length_first_arm + self.length_second_arm*np.cos(positive_rad_angle_second_arm)
        )
        positive_angle = positive_angle_part_1 - positive_angle_part_2

        negative_angle_part_1 = np.arctan2(y, x)
        negative_angle_part_2 = np.arctan2(
            self.length_second_arm * np.sin(negative_rad_angle_second_arm),
            self.length_first_arm + self.length_second_arm*np.cos(negative_rad_angle_second_arm)
        )
        negative_angle = negative_angle_part_1 - negative_angle_part_2

        if degrees:
            positive_angle *= (180/np.pi)
            positive_angle %= 360
            negative_angle *= (180/np.pi)
            negative_angle %= 360

        return positive_angle, negative_angle

    def _current_numeric_angles_second_arm(self, degrees=True) -> Tuple[float, float]:
        """
        Computes the two different numeric angles ('Elbow up' and 'Elbow down') of the second arm, using trigonometric
        inverse kinematics.

        Parameters
        ----------
        degrees : bool
            If the angle shall be returned as radians or degrees.

        Return
        ------
        float
            Current numeric angle of the second arm.
        """
        x = self.arm_route_x_y_positions[self.current_route_index][0]
        y = self.arm_route_x_y_positions[self.current_route_index][1]

        cos_angle = (x**2 + y**2 - self.length_first_arm**2 - self.length_second_arm**2) / \
                    (2*self.length_first_arm*self.length_second_arm)

        angle_positive = np.arctan2(
            +np.sqrt(1 - cos_angle**2),
            cos_angle
        )
        angle_negative = np.arctan2(
            -np.sqrt(1 - cos_angle**2),
            cos_angle
        )

        if degrees:
            angle_positive *= (180/np.pi)
            angle_positive %= 360
            angle_negative *= (180/np.pi)
            angle_negative %= 360

        return angle_positive, angle_negative

    def _num_step_first_motor(self) -> Tuple[int, int]:
        """
        Computes the direction and number of steps of first stepper motor to end up as close as possible to the
        current angle of the first arm.

        Return
        ------
        Tuple[int, int]
            Direction of arm ({-1 or 1}) and the number of steps to be taken.
        """
        diff_numeric_actual_first_arm = self.current_actual_angle_first_arm - self.previous_actual_angle_first_arm
        direction = 1 if diff_numeric_actual_first_arm >= 0 else -1
        num_steps = np.abs(diff_numeric_actual_first_arm//self.angle_per_motor_step)

        return direction, num_steps

    def _num_step_second_motor(self):
        """
        Computes the direction and number of steps of second stepper motor to end up as close as possible to the
        current angle of the second arm.

        Return
        ------
        Tuple[int, int]
            Direction of arm ({-1 or 1}) and the number of steps to be taken.
        """
        diff_numeric_actual_second_arm = self.current_actual_angle_second_arm - self.previous_actual_angle_second_arm
        direction = 1 if diff_numeric_actual_second_arm >= 0 else -1
        num_steps = np.abs(diff_numeric_actual_second_arm // self.angle_per_motor_step)

        return direction, num_steps

    def _compute_x_y_position_arms(self) -> Tuple[dict, dict]:
        """
        Compute the (x, y) coordinates of the start and end point of each arm.

        Return
        ------
        Tuple[dict, dict]
            First dict contains the (x, y) coordinates of the first arm. The second dict contains the (x, y)
            coordinates of the second arm.
        """
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

    def _init_function_visualization(self) -> Tuple[plt.Line2D, plt.Line2D, plt.Line2D]:
        """
        Initial function to start matplotlib's FuncAnimation.

        Return
        ------
        Tuple[plt.Line2D, plt.Line2D, plt.Line2D]
            Tuple containing the plots of first arm, second arm and trajectory.
        """
        x_y_first_arm, x_y_second_arm = self._compute_x_y_position_arms()

        self.first_arm_plot.set_data(x_y_first_arm['x'], x_y_first_arm['y'])
        self.second_arm_plot.set_data(x_y_second_arm['x'], x_y_second_arm['y'])
        self.x_y_trajectory_plot.set_data(list(self.x_trajectory), list(self.y_trajectory))

        return self.first_arm_plot, self.second_arm_plot, self.x_y_trajectory_plot

    def _visualization_step(self, i):
        """
        Step function in matplotlib's FuncAnimation.

        Parameters
        ----------
        i : int
            Animation counter. Not used.

        Return
        ------
        None
        """
        self._step()

        x_y_first_arm, x_y_second_arm = self._compute_x_y_position_arms()
        self.x_trajectory.append(x_y_second_arm['x'][1])
        self.y_trajectory.append(x_y_second_arm['y'][1])
        self.x_y_trajectory.append((x_y_second_arm['x'][1], x_y_second_arm['y'][1]))

        self.first_arm_plot.set_data(x_y_first_arm['x'], x_y_first_arm['y'])
        self.second_arm_plot.set_data(x_y_second_arm['x'], x_y_second_arm['y'])
        self.x_y_trajectory_plot.set_data(list(self.x_trajectory), list(self.y_trajectory))

    def _step(self):
        """
        Single step. Iterates to next (x, y) coordinate, computes new angles of arms and computes direction and number
        of steps for each arm.
        """
        self._update_current_numeric_angles()

        dir_first_arm, steps_first_arm = self._num_step_first_motor()
        dir_second_arm, steps_second_arm = self._num_step_second_motor()

        self.previous_actual_angle_first_arm = (self.previous_actual_angle_first_arm +
                                                dir_first_arm * steps_first_arm * self.angle_per_motor_step) % 360

        self.previous_actual_angle_second_arm = (self.previous_actual_angle_second_arm +
                                                 dir_second_arm * steps_second_arm * self.angle_per_motor_step) % 360

        if self.current_route_index < len(self.arm_route_x_y_positions) - 1:
            self.current_route_index += 1
        else:
            self.current_route_index = 0

    def start(self):
        animation_interval = len(self.arm_route_x_y_positions) - 1
        anim = FuncAnimation(self.fig, self._visualization_step, init_func=self._init_function_visualization,
                             interval=20, blit=False, save_count=animation_interval)
        #anim.save('line.gif', writer='imagemagick')
        plt.show()

    def stop(self):
        raise NotImplementedError
