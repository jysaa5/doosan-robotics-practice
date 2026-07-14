import rclpy
import DR_init
import sys
from robot import Robot
from config import ROBOT_ID, ROBOT_MODEL

def main(args=None):
    rclpy.init(args=args)


    DR_init.__dsr__id = ROBOT_ID
    DR_init.__dsr__model = ROBOT_MODEL

    node = rclpy.create_node("example_py", namespace=ROBOT_ID)
    DR_init.__dsr__node = node

    from DSR_ROBOT2 import set_robot_mode, ROBOT_MODE_AUTONOMOUS

    set_robot_mode(ROBOT_MODE_AUTONOMOUS)

    robot = Robot(node)

    gear_pick_point_1 = [362.04, -156.75, 26.35, 155.56, -179.95, 155.44]
    gear_pick_point_2 = [455.68, -205.70, 24.96, 159.39, 180.00, 159.10] 
    gear_pick_point_3 = [450.14, -100.40, 25.04, 169.66, 180.00, 165.24]

    gear_pick_up_1 = gear_pick_point_1.copy()
    gear_pick_up_2 = gear_pick_point_2.copy()
    gear_pick_up_3 = gear_pick_point_3.copy()

    gear_pick_up_1[2] += 100.00
    gear_pick_up_2[2] += 100.00
    gear_pick_up_3[2] += 100.00

    gear_place_point_1 = [365.69, 141.60, 24.80, 20.36, -180.00, 20.10]
    gear_place_point_2 = [450.09, 93.55, 25.13, 42.77, 179.96, 42.57]
    gear_place_point_3 = [454.46, 198.34, 26.24, 37.34, -179.97, 36.56]

    gear_place_up_1 = gear_place_point_1.copy()
    gear_place_up_2 = gear_place_point_2.copy()
    gear_place_up_3 = gear_place_point_3.copy()

    gear_place_up_1[2] += 100.00
    gear_place_up_2[2] += 100.00
    gear_place_up_3[2] += 100.00

    robot.home_position()
    robot.release()

    robot.move_l(gear_pick_up_1) # Bug of robot moveJ
    robot.move_l(gear_pick_up_1)
    robot.move_l(gear_pick_point_1)
    robot.grasp()
    robot.move_l(gear_pick_up_1)
    robot.move_l(gear_place_up_1)
    robot.move_l(gear_place_point_1)
    robot.release()
    robot.move_l(gear_place_up_1)

    robot.move_l(gear_pick_up_2)
    robot.move_l(gear_pick_point_2)
    robot.grasp()
    robot.move_l(gear_pick_up_2)
    robot.move_l(gear_place_up_2)
    robot.move_l(gear_place_point_2)
    robot.release()
    robot.move_l(gear_place_up_2)

    robot.move_l(gear_pick_up_3)
    robot.move_l(gear_pick_point_3)
    robot.grasp()
    robot.move_l(gear_pick_up_3)
    robot.move_l(gear_place_up_3)
    robot.move_l(gear_place_point_3)
    robot.release()
    robot.move_l(gear_place_up_3)

    robot.home_position()

    print("Example complete")
    rclpy.shutdown()

if __name__ == "__main__":
    main()
