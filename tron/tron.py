import ast
import sys
import time
import threading
from functools import partial
import limxsdk.robot.Rate as Rate
import limxsdk.robot.Robot as Robot
import limxsdk.robot.RobotType as RobotType
import limxsdk.datatypes as datatypes

user_input = ""
lock = threading.Lock()

def user_input_handler(input_text, func):
    user_input = input(input_text)
    if user_input.strip() == "":
        return
    func(user_input)

def make_setter(target_dict, key):
    return lambda value: target_dict.__setitem__(key, ast.literal_eval(value))

class RobotReceiver:
    def __init__(self, motor_number):
        self.imu_data = {}
        self.robot_state = {}
        self.sensor_joy = {}
        self.diagnostic_value = {}
        self.motor_number = motor_number
        self.robot_command = {}

        self.clear_robot_command()

    def clear_robot_command(self):
        # Hardcode values for sole-foot config
        self.robot_command['mode'] = [1.0 for _ in range(self.motor_number)]
        self.robot_command['q'] = [1.0 for _ in range(self.motor_number)]
        self.robot_command['dq'] = [1.0 for _ in range(self.motor_number)]
        self.robot_command['tau'] = [1.0 for _ in range(self.motor_number)]
        self.robot_command['Kp'] = [1.0 for _ in range(self.motor_number)]
        self.robot_command['Kd'] = [1.0 for _ in range(self.motor_number)]

    # Callback function for receiving imu
    def imuDataCallback(self, imu: datatypes.ImuData):
        self.imu_data['stamp'] = imu.stamp
        self.imu_data['acc'] = imu.acc
        self.imu_data['gyro'] = imu.gyro
        self.imu_data['quat'] = imu.quat

    # Callback function for receiving robot state
    def robotStateCallback(self, robot_state: datatypes.RobotState):
        self.robot_state['stamp'] = robot_state.stamp
        self.robot_state['tau'] = robot_state.tau
        self.robot_state['q'] = robot_state.q
        self.robot_state['dq'] = robot_state.dq

    # Callback function for receiving sensor joy data
    def sensorJoyCallback(self, sensor_joy: datatypes.SensorJoy):
        self.sensor_joy['stamp'] = sensor_joy.stamp
        self.sensor_joy['axes'] = sensor_joy.axes
        self.sensor_joy['buttons'] = sensor_joy.buttons

    # Callback function for receiving diagnostic value
    def diagnosticValueCallback(self, diagnostic_value: datatypes.DiagnosticValue):
        self.diagnostic_value['stamp'] = diagnostic_value.stamp
        self.diagnostic_value['name'] = diagnostic_value.name
        self.diagnostic_value['level'] = diagnostic_value.level
        self.diagnostic_value['code'] = diagnostic_value.code
        self.diagnostic_value['message'] = diagnostic_value.message

    def user_input_loop(self):
        global user_input
        while True:
            with lock:
                user_input = input("Input command (s|stop, imu, robot_state, sensor_joy, diagnostic, robot_command)\n")

            if user_input == "imu":
                if not self.imu_data:
                    "No imu_data has been received!"
                else:
                    print("\n------\nLast ImuData:" + \
                        "\n  stamp: " + str(self.imu_data['stamp']) + \
                        "\n  acc: " + str(self.imu_data['acc']) + \
                        "\n  gyro: " + str(self.imu_data['gyro']) + \
                        "\n  quat: " + str(self.imu_data['quat']))
            elif user_input == "robot_state":
                if not self.robot_state:
                    "No robot_state has been received!"
                else:
                    print("\n------\nLast RobotState:" + \
                        "\n  stamp: " + str(self.robot_state['stamp']) + \
                        "\n  tau: " + str(self.robot_state['tau']) + \
                        "\n  q: " + str(self.robot_state['q']) + \
                        "\n  dq: " + str(self.robot_state['dq']))
            elif user_input == "sensor_joy":
                if not self.sensor_joy:
                    "No sensor_joy has been received!"
                else:
                    print("\n------\nLast sensor_joy:" + \
                        "\n  stamp: " + str(self.sensor_joy['stamp']) + \
                        "\n  axes: " + str(self.sensor_joy['axes']) + \
                        "\n  buttons: " + str(self.sensor_joy['buttons']))
            elif user_input == "diagnostic":
                if not self.diagnostic_value:
                    "No diagnostic_value has been received!"
                else:
                    print("\n------\nLast diagnostic_value:" + \
                        "\n  stamp: " + str(self.diagnostic_value['stamp']) + \
                        "\n  name: " + self.diagnostic_value['name'] + \
                        "\n  level: " + str(self.diagnostic_value['level']) + \
                        "\n  code: " + str(self.diagnostic_value['code']) + \
                        "\n  message: " + self.diagnostic_value['message'])
            elif user_input == "robot_command":
                if not self.robot_state:
                    "No robot_state has been received!"
                else:
                    print("\n------\nLast RobotState:" + \
                        "\n  stamp: " + str(self.robot_state['stamp']) + \
                        "\n  tau: " + str(self.robot_state['tau']) + \
                        "\n  q: " + str(self.robot_state['q']) + \
                        "\n  dq: " + str(self.robot_state['dq']))
                print("\n------\nenter robot_command as prompted (blank for default):")
                user_input_handler("Enter mode: ", make_setter(self.robot_command, "mode"))
                user_input_handler("Enter q: ", make_setter(self.robot_command, "q"))
                user_input_handler("Enter dq: ", make_setter(self.robot_command, "dq"))
                user_input_handler("Enter tau: ", make_setter(self.robot_command, "dq"))
                user_input_handler("Enter Kp: ", make_setter(self.robot_command, "Kp"))
                user_input_handler("Enter Kd: ", make_setter(self.robot_command, "Kd"))
            elif user_input == "stop" or user_input == "s":
                self.clear_robot_command()

if __name__ == '__main__':
    # Create a Robot instance
    # Options: PointFoot, Wheellegged, Humanoid
    robot = Robot(RobotType.PointFoot)

    # print("Available RobotType values:")
    # for name in dir(RobotType):
    #     if not name.startswith("_"):
    #         value = getattr(RobotType, name)
    #         print(f"- {name} = {value}")

    robot_ip = "10.192.1.2"
    # Check if command-line argument is provided for robot IP
    if len(sys.argv) > 1:
        robot_ip = sys.argv[1]

    # Initialize the robot with robot_ip
    if not robot.init(robot_ip):
        sys.exit()

    # Get motor number information
    motor_number = robot.getMotorNumber()
    print(f"Number of motors: {motor_number}")

    # Create an instance of RobotReceiver to handle callbacks
    receiver = RobotReceiver(motor_number)

    # Create partial functions for callbacks
    imuDataCallback = partial(receiver.imuDataCallback)
    robotStateCallback = partial(receiver.robotStateCallback)
    sensorJoyCallback = partial(receiver.sensorJoyCallback)
    diagnosticValueCallback = partial(receiver.diagnosticValueCallback)

    # Subscribe to robot state, sensor joy, and diagnostic value topics
    robot.subscribeImuData(imuDataCallback)
    robot.subscribeRobotState(robotStateCallback)
    robot.subscribeSensorJoy(sensorJoyCallback)
    robot.subscribeDiagnosticValue(diagnosticValueCallback)

    threading.Thread(target=receiver.user_input_loop, daemon=True).start()

    # Main loop to continuously publish robot commands
    rate = Rate(1000) # 1000 Hz
    while True:
        cmd_msg = datatypes.RobotCmd()
        cmd_msg.stamp = time.time_ns()  # Set the timestamp
        # Set default values for control mode, joint positions, velocities, torques, Kp, and Kd
        cmd_msg.mode = receiver.robot_command['mode']
        cmd_msg.q = receiver.robot_command['q']
        cmd_msg.dq = receiver.robot_command['dq']
        cmd_msg.tau = receiver.robot_command['tau']
        cmd_msg.Kp = receiver.robot_command['Kp']
        cmd_msg.Kd = receiver.robot_command['Kd']
        # print(f"motor_number: {receiver.motor_number}, len(cmd_msg.q): {len(cmd_msg.q)}")
        robot.publishRobotCmd(cmd_msg)  # Publish the robot command
        rate.sleep()  # Control loop frequency
