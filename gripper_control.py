import rclpy
import DR_init
import sys
import time
from config import ROBOT_ID, ROBOT_MODEL

def main(args=None):
    rclpy.init(args=args)


    DR_init.__dsr__id = ROBOT_ID
    DR_init.__dsr__model = ROBOT_MODEL

    node = rclpy.create_node("example_py", namespace=ROBOT_ID)

    DR_init.__dsr__node = node

    from DSR_ROBOT2 import set_tool_digital_output, set_robot_mode, ROBOT_MODE_AUTONOMOUS

    set_robot_mode(ROBOT_MODE_AUTONOMOUS)

    def grasp():
        set_tool_digital_output(index=1, val=1)
        set_tool_digital_output(index=2, val=0)
        time.sleep(1)

    def release():
        set_tool_digital_output(index=2, val=1)
        set_tool_digital_output(index=1, val=0)
        time.sleep(1)

    while True:
        grasp()
        release()

    print("Example complete")
    rclpy.shutdown()

if __name__ == "__main__":
    main()