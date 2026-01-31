import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/roberd/ros2_ws/src/aruco_py/aruco_py/install/aruco_py'
