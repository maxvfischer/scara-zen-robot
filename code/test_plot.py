from utils import ScaraRobot

if __name__ == "__main__":
    robot = ScaraRobot(machine_config_path='machine_config.json')

    robot.start()
