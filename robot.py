import rclpy
import DR_init
import sys
import time

class Robot():
    def __init__(self, node):
        self.node = node

        from DSR_ROBOT2 import movel, movej, set_tool_digital_output, move_home
        self.movel = movel
        self.movej = movej
        self.set_tool_digital_output = set_tool_digital_output
        self.move_home = move_home

    def move_l(self, task):
        self.movel(task, vel=50, acc=50)

    def move_j(self, joint):
        self.movej(joint, vel=25, acc=25)

    def home_position(self):
        self.move_home(1)

    def grasp(self):
        self.set_tool_digital_output(index=1, val=1)
        self.set_tool_digital_output(index=2, val=0)
        time.sleep(1)

    def release(self):
        self.set_tool_digital_output(index=2, val=1)
        self.set_tool_digital_output(index=1, val=0)
        time.sleep(1)