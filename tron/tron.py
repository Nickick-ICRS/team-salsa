import sys
import limxsdk.robot.Robot as Robot
import limxsdk.robot.RobotType as RobotType

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

    # Get the number of motors in the robot
    motor_number = robot.getMotorNumber()
