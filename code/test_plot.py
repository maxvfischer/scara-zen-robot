from utils import ScaraRobot
from utils import FLOWER_1, FLOWER_2, SPIRAL_1, HAMPA_1, ROSE, BUTTERFLY_1
from utils import SCARA_ROBOT_V1

if __name__ == "__main__":
    robot = ScaraRobot(
        length_first_arm=SCARA_ROBOT_V1['length_first_arm'],
        length_second_arm=SCARA_ROBOT_V1['length_second_arm'],
        angle_per_motor_step=SCARA_ROBOT_V1['angle_per_motor_step'],
        timeout_between_steps_sec=SCARA_ROBOT_V1['timeout_between_steps_sec'],
        art_config=FLOWER_2,
        visualization_mode=True,
        number_of_steps=1000
    )

    robot.start()
