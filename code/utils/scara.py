import os
import json
import copy
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from typing import Callable, List, Tuple, Union
from matplotlib.animation import FuncAnimation
from google.cloud import firestore


class ScaraRobot:
    """
    SCARA robot using two arms.

    See the paper "Formal Kinematic Analysis of the Two-Link Planar Manipulator"
    (https://www.researchgate.net/publication/283377443_Formal_Kinematic_Analysis_of_the_Two-Link_Planar_Manipulator)
    for inverse kinematic equations used.

    Arguments
    ---------
    machine_config_path : str
        Path to the machine configuration file, containing the SCARA robot uid.
    """
    def __init__(self,
                 machine_config_path: str):
        if not os.path.isfile(machine_config_path):
            raise FileExistsError(f"Config file '{machine_config_path}' does not exist.")
        else:
            with open(machine_config_path) as f:
                machine_config = json.load(f)
        self.machine_id = machine_config['machine_id']

        # Machine setup
        self.length_first_arm = None
        self.length_second_arm = None
        self.angle_per_motor_step = None

        # Artwork settings
        self.user_stop_scara_arm = False
        self.number_of_steps_to_origo = None
        self.current_route_index = 0
        self.current_artwork_index = 0
        self.active_artwork_patterns = []

        # Firebase setup
        self.db = self._initialize_firestore_database(
            gcloud_key_path=machine_config['gcloud_key_path']
        )
        self.firebase_watcher = self._fetch_and_listen_to_user_setting_changes()
        time.sleep(3)

        # SCARA robot variables
        self.current_route_index = 0
        self.current_actual_angle_first_arm = 90.
        self.current_actual_angle_second_arm = 180.
        self.previous_actual_angle_first_arm = 90.
        self.previous_actual_angle_second_arm = 180.

        # Plotting settings
        plt.style.use("ggplot")
        absolute_value_axis_limits = self.length_first_arm + self.length_second_arm
        self.fig = plt.figure()
        self.ax = plt.axes(
            xlim=(-absolute_value_axis_limits, absolute_value_axis_limits),
            ylim=(-absolute_value_axis_limits, absolute_value_axis_limits)
        )

        x_y_first_arm, x_y_second_arm = self._compute_x_y_position_arms()
        self.x_y_trajectory = deque([(x_y_second_arm['x'][1], x_y_second_arm['y'][1])])
        self.x_trajectory = deque([x_y_second_arm['x'][1]],
                                  maxlen=1000)
        self.y_trajectory = deque([x_y_second_arm['y'][1]],
                                  maxlen=1000)

        self.first_arm_plot, = self.ax.plot(x_y_first_arm['x'], x_y_first_arm['y'])
        self.second_arm_plot, = self.ax.plot(x_y_second_arm['x'], x_y_second_arm['y'])
        self.x_y_trajectory_plot, = self.ax.plot(x_y_second_arm['x'][1], x_y_second_arm['y'][1])

    @staticmethod
    def _initialize_firestore_database(gcloud_key_path: str):
        """
        Initialize a connection with Firebase's Firestore. Credentials are fetched from the environment variable
        GOOGLE_APPLICATION_CREDENTIALS.

        Parameters
        ----------
        gcloud_key_absolute_path : str
            Absolute path to gcloud service account key

        Return
        ------
        firestore.Client
            Firebase
        """
        assert os.path.isfile(gcloud_key_path), \
            "gcloud service auth key could not be found"

        return firestore.Client.from_service_account_json(gcloud_key_path)

    def _fetch_and_listen_to_user_setting_changes(self):
        """
        Setting up a watcher on the user_settings for this specific machine's uid. The function
        self._on_firebase_change_user_settings will be executed on instantiation of this class, as well as when
        a change has happened on firestore to the 'user_settings' for this specific machine.

        Return
        ------
        firestore.Watch
            Firestore watcher
        """
        return self.db.collection(u'user_settings') \
                      .document(self.machine_id) \
                      .on_snapshot(self._on_firebase_change_user_settings)

    def _on_firebase_change_user_settings(self, docs, changes, read_time):
        """
        Updating the robot and artwork settings when a change has happened on firestore, for this specific
        machine's uid.

        If the machine is running an there's an active (x, y) arm route, the machine will return to origo and pause
        the scara arm before updating the settings.
        """
        for doc in docs:
            user_settings = doc.to_dict()
            if self.length_first_arm != user_settings['length_first_arm']:
                self.length_first_arm = user_settings['length_first_arm']
            if self.length_second_arm != user_settings['length_second_arm']:
                self.length_second_arm = user_settings['length_second_arm']
            if self.angle_per_motor_step != user_settings['angle_per_motor_step']:
                self.angle_per_motor_step = user_settings['angle_per_motor_step']
            if self.number_of_steps_to_origo != user_settings['number_of_steps_to_origo']:
                self.number_of_steps_to_origo = user_settings['number_of_steps_to_origo']
            if self.user_stop_scara_arm != user_settings['stop_scara_arm']:
                self.user_stop_scara_arm = user_settings['stop_scara_arm']

            firebase_artwork_patterns = set()
            for pattern_document in user_settings['active_artwork_patterns']:
                firebase_artwork_patterns.add(pattern_document.id)
            local_artwork_patterns = [artwork['name'] for artwork in self.active_artwork_patterns]
            local_artwork_patterns = set(local_artwork_patterns)

            # Artwork removed
            if firebase_artwork_patterns < local_artwork_patterns:
                removed_artwork_names = local_artwork_patterns - firebase_artwork_patterns
                currently_displayed_artwork_name = self.active_artwork_patterns[self.current_artwork_index]['name']
                if currently_displayed_artwork_name in removed_artwork_names:
                    x_y_route_to_origo = self._route_to_origo_x_y()
                    self.active_artwork_patterns.append(
                        {
                            'name': 'route_to_origo',
                            'arm_route_x_y_positions': x_y_route_to_origo,
                        }
                    )

                    idx_to_remove = [idx
                                     for idx, pattern in enumerate(self.active_artwork_patterns)
                                     if pattern['name'] == currently_displayed_artwork_name][0]
                    del self.active_artwork_patterns[idx_to_remove]

                    self.current_route_index = 0
                    self.current_artwork_index = len(self.active_artwork_patterns) - 1

                else:
                    idx_to_remove = [idx
                                     for idx, pattern in enumerate(self.active_artwork_patterns)
                                     if pattern['name'] in removed_artwork_names]
                    for index in sorted(idx_to_remove, reverse=True):
                        del self.active_artwork_patterns[index]
                        if self.current_artwork_index > index:
                            self.current_artwork_index -= 1

            # Artwork added
            elif firebase_artwork_patterns > local_artwork_patterns:
                added_artwork_names = firebase_artwork_patterns - local_artwork_patterns
                for pattern_document in user_settings['active_artwork_patterns']:
                    if pattern_document.id in added_artwork_names:
                        artwork_pattern_settings = pattern_document.get().to_dict()
                        x_y_route = self._generate_x_y_artwork_pattern(
                            lambda_function=artwork_pattern_settings['polar_equation'],
                            lower_limit=artwork_pattern_settings['lower_limit'],
                            upper_limit=artwork_pattern_settings['upper_limit'],
                            scale=artwork_pattern_settings['scale'],
                            num_steps=artwork_pattern_settings['number_of_steps_per_cycle']
                        )
                        self.active_artwork_patterns.append(
                            {
                                'name': pattern_document.id,
                                'arm_route_x_y_positions': x_y_route,
                                'steps_to_origo': 0
                            }
                        )

    @staticmethod
    def _artwork_is_cyclic(route: List[Tuple[float, float]], epsilon=1e-6):
        """
        Checks if artwork is cyclic by checking if the distance between (x_first, y_first) and (x_last, y_last) is
        smaller than epsilon.

        Parameters
        ----------
        route : List[Tuple[float, float]]
            List of (x, y) coordinates of the artwork.

        epsilon : float
            Epsilon to compare distance with.

        Return
        ------
        bool
            If artwork is cyclic or not.

        """
        first_x, first_y = route[0][0], route[0][1]
        last_x, last_y = route[-1][0], route[-1][1]
        distance_first_to_last = np.sqrt((last_x-first_x)**2 + (last_y-first_y)**2)

        if distance_first_to_last < epsilon:
            return True
        else:
            return False

    def _generate_x_y_artwork_pattern(self,
                                      lambda_function: Union[str, Callable],
                                      num_steps: Union[int, str],
                                      lower_limit: Union[str, float],
                                      upper_limit: Union[float, str],
                                      scale: Union[str, float] = 1.) -> List[Tuple[float, float]]:
        """
        Generates a list of (x, y) coordinates, generated from a polar equation implemented as a lambda function.

        Parameters
        ----------
        lambda_function : str or Callable
            Radius polar equation implemented as a lambda function. The lambda function is only allowed to include
            the parameter "theta", and shall return a single float or integer.

            Example: lambda_function = eval("lambda theta: 3 + np.sin(10*theta)") or
            "lambda theta: 3 + np.sin(10*theta)"

        num_steps : str or int
            Number of samples to generate between lower_limit and upper_limit.

        lower_limit : str or float
            Starting value in the list of theta.

        upper_limit : str or float
            Ending value in the list of theta.

        scale : str or float
            Scaling factor. Increases size of polar equation graph.

        Return
        ------
        list
            List of tuples, containing (x, y) coordinates.
        """
        if isinstance(lambda_function, str):
            lambda_function = eval(lambda_function)
        if isinstance(num_steps, str):
            num_steps = int(num_steps)
        if isinstance(scale, str):
            scale = float(scale)
        if isinstance(lower_limit, str):
            lower_limit = eval(lower_limit)
        if isinstance(upper_limit, str):
            upper_limit = eval(upper_limit)

        import inspect
        lambda_args = inspect.getfullargspec(lambda_function).args
        assert len(lambda_args) == 1 and lambda_args[0] == 'theta', \
            "lambda_function is only allowed to have one parameter: 'theta'"

        # Generate (x, y) coordinates of artwork polar equation
        theta_list = np.linspace(lower_limit, upper_limit, num_steps)
        coordinates = []
        for theta in theta_list:
            r = lambda_function(theta=theta)*scale
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            coordinates.append((x, y))

        # Check if (x, y) coordinates are OK for arms to draw, i.e. it's long enought to reach the (x, y)
        # positions.
        for x, y in coordinates:
            if np.sqrt(x**2 + y**2) > self.length_first_arm + self.length_second_arm:
                raise ValueError(f"Magnitude from origo to ({x}, {y}) too large for arms to draw. "
                             f"Adjust polar equation.")

        # If artwork is NOT cyclic, i.e. (last_x, last_y) !≈ (first_x, first_y),
        # generate mirrored route and extend at the end of the list of coordinates.
        # Artwork is now cyclic, i.e. (last_x, last_y) ≈ (first_x, first_y)
        if not self._artwork_is_cyclic(route=coordinates):
            route = copy.copy(coordinates)
            last_x, last_y = route[-1][0], route[-1][1]
            if np.abs(last_x) < np.abs(last_y):
                route = [(-1*x, y) for x, y in route]
            else:
                route = [(x, -1*y) for x, y in route]
            mirrored_route = np.flip(route, axis=0)
            coordinates.extend(mirrored_route)

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
        x = self.active_artwork_patterns[self.current_artwork_index]['arm_route_x_y_positions'][self.current_route_index][0]
        y = self.active_artwork_patterns[self.current_artwork_index]['arm_route_x_y_positions'][self.current_route_index][1]

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
        x = self.active_artwork_patterns[self.current_artwork_index]['arm_route_x_y_positions'][self.current_route_index][0]
        y = self.active_artwork_patterns[self.current_artwork_index]['arm_route_x_y_positions'][self.current_route_index][1]

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

    def _num_step_second_motor(self) -> Tuple[int, int]:
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

    def _route_to_origo_x_y(self, num_steps=50) -> List[Tuple[float, float]]:
        """
        Compute (x, y) coordinates of a straight route from the current (x, y) coordinate to origo.

        TODO: Most artwork patterns start of in a horizontal direction from origo, thus starting with
              the first arm on the y-axis (angle 90 or 270 degrees) and the second arm folded on the
              first arm (angle 180 degrees). When returning to origo, the angles of the arms usually
              end up somewhere else when arriving at origo. Sometimes this lead to the need of taking
              a couple of hundred steps of first arm to end up on the y-axis.
              If this is a problem, two potential solutions:
              1) Choose the "elbow up" or "elbow down" that moves the first arm closest to the y-axis.
              2) Rotate all artwork pattern routes (rotate the coordinate system), so the first arm
                 ends up at the y-axis.


        Parameters
        ----------
        num_steps : int
            Number of steps from current (x, y) coordinate to origo.

        Returns
        -------
        List[Tuple[float, float]]
            List of tuples of (x, y) coordinates.
        """
        _, x_y_second_arm = self._compute_x_y_position_arms()
        current_x, current_y = x_y_second_arm['x'][1], x_y_second_arm['y'][1]

        if current_x >= 0:
            x, y = (0, current_x), (0, current_y)
            x_interpolation = np.linspace(x[0], x[1], num_steps)
            y_interpolation = np.interp(x_interpolation, x, y)
            x_interpolation = np.flip(x_interpolation)
            y_interpolation = np.flip(y_interpolation)
        else:
            x, y = (current_x, 0), (current_y, 0)
            x_interpolation = np.linspace(x[0], x[1], num_steps)
            y_interpolation = np.interp(x_interpolation, x, y)

        route_to_origo = [(x_coordinate, y_coordinate)
                          for x_coordinate, y_coordinate in zip(x_interpolation, y_interpolation)]

        return route_to_origo

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
        if (not self.user_stop_scara_arm) and (len(self.active_artwork_patterns) != 0):
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

        arm_route_x_y_positions = self.active_artwork_patterns[self.current_artwork_index]['arm_route_x_y_positions']
        name_of_current_artwork = self.active_artwork_patterns[self.current_artwork_index]['name']
        if self.current_route_index < len(arm_route_x_y_positions) - 1:
            self.current_route_index += 1
        else:
            self.current_route_index = 0
            if self.current_artwork_index < len(self.active_artwork_patterns) - 1:
                self.current_artwork_index += 1
            else:
                self.current_artwork_index = 0
            if name_of_current_artwork == 'route_to_origo':
                del self.active_artwork_patterns[-1]

    def start(self):
        anim = FuncAnimation(self.fig, self._visualization_step, init_func=self._init_function_visualization,
                             interval=100, blit=False)
        #anim.save('line.gif', writer='imagemagick')
        plt.show()

    def stop(self):
        raise NotImplementedError
