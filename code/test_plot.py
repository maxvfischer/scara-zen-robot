from utils import ScaraRobot
from utils import (FLOWER_1, FLOWER_2, FLOWER_3, SPIRAL_1, SPIRAL_2, HAMPA_1,
                   ERASE_1)
from utils import SCARA_ROBOT_V1

if __name__ == "__main__":
    robot = ScaraRobot(
        length_first_arm=SCARA_ROBOT_V1['length_first_arm'],
        length_second_arm=SCARA_ROBOT_V1['length_second_arm'],
        angle_per_motor_step=SCARA_ROBOT_V1['angle_per_motor_step'],
        art_config=ERASE_1,
        visualization_mode=True,
        number_of_steps=1000
    )

    robot.start()
