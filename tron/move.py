import sys
import time
import limxsdk.robot.Rate as Rate
import limxsdk.robot.Robot as Robot
import limxsdk.robot.RobotType as RobotType
import limxsdk.datatypes as datatypes

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

    # Get information about joint offset, joint limit, and motor number
    joint_offset = robot.getJointOffset()
    joint_limit = robot.getJointLimit()
    motor_number = robot.getMotorNumber()
    
    # Main loop to continuously publish robot commands
    rate = Rate(500) # 1500 Hz
    cmd_msg = datatypes.RobotCmd()
    while True:
        # Set default values for the timestamp, control mode, joint position, speed, torque, Kp, and Kd
        cmd_msg.stamp = time.time_ns()
        cmd_msg.mode = [1.0 for _ in range(motor_number)]
        cmd_msg.q = [1.0 for _ in range(motor_number)]
        cmd_msg.dq = [1.0 for _ in range(motor_number)]
        cmd_msg.tau = [1.0 for _ in range(motor_number)]
        cmd_msg.Kp = [1.0 for _ in range(motor_number)]
        cmd_msg.Kd = [1.0 for _ in range(motor_number)]
        robot.publishRobotCmd(cmd_msg)  # Publish robot command
        rate.sleep()  # Control loop frequency