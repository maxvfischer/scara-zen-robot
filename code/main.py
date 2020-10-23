from .utils import ScaraRobot

if __name__ == "__main__":
    robot = ScaraRobot(machine_config_path='code/credentials/machine_config.json')

    robot.start()
