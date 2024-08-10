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
        # ��ROSͼ����Ϣת��ΪOpenCVͼ��
        cv_img = self.br.imgmsg_to_cv2(msg, "bgr8")

        # ��ͼ�����Ͻǻ��ƾ���
        height, width, _ = cv_img.shape
        cv2.rectangle(cv_img, (width - 110, 10), (width - 10, 110), (0, 255, 0), 3)

        # ��OpenCVͼ��ת����ROSͼ����Ϣ
        ros_image = self.br.cv2_to_imgmsg(cv_img, encoding="bgr8")

        # ����������ͼ��
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
