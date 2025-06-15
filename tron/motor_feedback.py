import sys
from functools import partial
import limxsdk.robot.Robot as Robot
import limxsdk.robot.RobotType as RobotType
import limxsdk.datatypes as datatypes

class RobotReceiver:
    # Callback function to receive the robot state
    def robotStateCallback(self, robot_state: datatypes.RobotState):
        print("\n------\nrobot_state:" + \
              "\n  stamp: " + str(robot_state.stamp) + \
              "\n  tau: " + str(robot_state.tau) + \
              "\n  q: " + str(robot_state.q) + \
              "\n  dq: " + str(robot_state.dq))

if __name__ == '__main__':
    # Create a Robot instance of type PointFoot
    robot = Robot(RobotType.PointFoot)

    robot_ip = "10.192.1.2"
    # Check if the command-line argument is provided as the robot IP
    if len(sys.argv) > 1:
        robot_ip = sys.argv[1]

    # Use robot_ip to initialize the robot
    if not robot.init(robot_ip):
        sys.exit()

    # Create a RobotReceiver instance to process the callback
    receiver = RobotReceiver()

    # Create a partial function for the callback function
    robotStateCallback = partial(receiver.robotStateCallback)

    # Subscribe to the robot state
    robot.subscribeRobotState(robotStateCallback)
    
    # Sleep for 1 second to prevent program exit
    import time
    while True:
        time.sleep(1) 