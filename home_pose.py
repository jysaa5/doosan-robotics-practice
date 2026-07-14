import rclpy
import DR_init
import sys
from config import ROBOT_ID, ROBOT_MODEL

def main(args=None):
	rclpy.init(args=args)
	
	DR_init.__dsr__id = ROBOT_ID
	DR_init.__dsr__model = ROBOT_MODEL
	
	node = rclpy.create_node('example_py', namespace=ROBOT_ID)
	DR_init.__dsr__node = node
	
	from DSR_ROBOT2 import movej, set_robot_mode, ROBOT_MODE_AUTONOMOUS
	
	set_robot_mode(ROBOT_MODE_AUTONOMOUS)
	
	home_joint = [0.00, 0.00, 90.00, 0.00, 90.00, 0.00]
	movej(home_joint, vel=25, acc=25)
	print(f"@@@@@@@")
	
	print("Example complete")
	rclpy.shutdown()
if __name__ == '__main__':
	main()
