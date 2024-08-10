import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class ImageProcessor(Node):

    def __init__(self):
        super().__init__('image_processor')
        self.subscription = self.create_subscription(
            Image,
            'image_raw',
            self.listener_callback,
            10)
        self.publisher_ = self.create_publisher(Image, 'processed_image', 10)
        self.br = CvBridge()

    def listener_callback(self, msg):
        # 将ROS图像消息转换为OpenCV图像
        cv_img = self.br.imgmsg_to_cv2(msg, "bgr8")

        # 在图像右上角绘制矩形
        height, width, _ = cv_img.shape
        cv2.rectangle(cv_img, (width - 110, 10), (width - 10, 110), (0, 255, 0), 3)

        # 将OpenCV图像转换回ROS图像消息
        ros_image = self.br.cv2_to_imgmsg(cv_img, encoding="bgr8")

        # 发布处理后的图像
        self.publisher_.publish(ros_image)
        self.get_logger().info('Publishing processed image')

def main(args=None):
    rclpy.init(args=args)
    image_processor = ImageProcessor()
    rclpy.spin(image_processor)
    image_processor.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
