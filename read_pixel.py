import cv2

def mouse_move_callback(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        print(f"Mouse point: ({x:4d}, {y:4d})", end="\r")

img = cv2.imread("/home/hyu/ros2_ws/src/doosan-robot2/output.png")

if img is None:
    print("Unable to mount image. Check your image and path")
else:
    window_name = "Mouse Tracking"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_move_callback)

    print("Move your mouse over the image window. Press any key if you want to exit")
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()