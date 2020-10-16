from utils import ScaraRobot
from utils import CIRCLE, FLOWER, FLOWER_2

if __name__ == "__main__":
    robot = ScaraRobot(
        length_first_arm=7,
        length_second_arm=7,
        angle_per_motor_step=365/4094,
        timeout_between_steps_sec=0.2,
        arm_route_x_y_positions=FLOWER_2,
        visualization_mode=True
    )

    for i in range(0, 200):
        robot.start()
