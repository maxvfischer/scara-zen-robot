from utils import ScaraRobot

if __name__ == "__main__":
    robot = ScaraRobot(
        config_dir='scara_config.json',
    )

    robot.start()
