import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/gg/ros2_ws/src/my_image_processing_pkg/install/my_image_processing_pkg'
