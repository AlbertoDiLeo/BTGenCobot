#!/usr/bin/env python3
"""Florence-2 Object Detection Service Node

This ROS2 service node provides object detection using:
- Florence-2 for text-prompted object detection (VLM-based, understands attributes)
- Bounding box sampling for depth/pose estimation

Service: /detect_object (btgencobot_interfaces/srv/DetectObject)
"""

import rclpy
from rclpy.node import Node
from btgencobot_interfaces.srv import DetectObject
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2
import time

try:
    import torch
    from transformers import AutoProcessor, AutoModelForCausalLM
    from PIL import Image as PILImage
    DEPENDENCIES_AVAILABLE = True
    import_error = None
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    import_error = str(e)


class Florence2Service(Node):
    """ROS2 service node for Florence-2 object detection with bounding box pose estimation"""

    def __init__(self):
        super().__init__('florence2_service')
        self._declare_parameters()
        self._setup_device()
        self._initialize_models()
        self._create_service()

    def _declare_parameters(self):
        """Declare and load ROS parameters"""
        self.declare_parameter('use_mock', False)
        self.declare_parameter('florence2_model', 'microsoft/Florence-2-base-ft')
        self.declare_parameter('device', 'auto')
        self.declare_parameter('publish_debug_images', True)

        self.use_mock = self.get_parameter('use_mock').value
        self.florence2_model_name = self.get_parameter('florence2_model').value
        self.device_param = self.get_parameter('device').value
        self.publish_debug_images = self.get_parameter('publish_debug_images').value

    def _setup_device(self):
        """Setup compute device (CUDA or CPU)"""
        if self.device_param == 'auto':
            self.device = 'cuda' if (DEPENDENCIES_AVAILABLE and torch.cuda.is_available()) else 'cpu'
        else:
            self.device = self.device_param

        self.get_logger().info('Florence-2 Service Node starting...')
        self.get_logger().info(f'Use mock: {self.use_mock}')
        self.get_logger().info(f'Device: {self.device}')

    def _initialize_models(self):
        """Initialize Florence-2 model or fallback to mock mode"""
        self.florence2_model = None
        self.florence2_processor = None
        self.bridge = CvBridge()

        if not self.use_mock:
            if not DEPENDENCIES_AVAILABLE:
                self.get_logger().error(f'Dependencies not available: {import_error}')
                self.get_logger().warning('Falling back to MOCK mode')
                self.use_mock = True
            else:
                self._load_models()

        if self.use_mock:
            self.get_logger().warning('Running in MOCK MODE - will return fake detections')

    def _load_models(self):
        """Load Florence-2 model"""
        try:
            # Load Florence-2
            self.get_logger().info('Loading Florence-2 model...')
            self.get_logger().info(f'Model: {self.florence2_model_name}')

            torch_dtype = torch.float16 if self.device == 'cuda' else torch.float32
            self.florence2_model = AutoModelForCausalLM.from_pretrained(
                self.florence2_model_name,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                attn_implementation='eager'
            ).to(self.device)
            self.florence2_processor = AutoProcessor.from_pretrained(
                self.florence2_model_name,
                trust_remote_code=True
            )
            self.get_logger().info('Florence-2 model loaded successfully')

        except Exception as e:
            self.get_logger().error(f'Failed to load models: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())
            self.get_logger().warning('Falling back to MOCK mode')
            self.use_mock = True

    def _create_service(self):
        """Create ROS2 service and debug image publisher"""
        self.service = self.create_service(
            DetectObject,
            '/detect_object',
            self.detect_callback
        )
        self.debug_image_pub = self.create_publisher(Image, '/florence2/debug_image', 10)
        self.get_logger().info('Service /detect_object ready')

    def detect_callback(self, request, response):
        """Handle detection service request"""
        try:
            start_time = time.time()
            
            # Convert ROS Image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(request.image, desired_encoding='rgb8')
            self.get_logger().info(
                f'Detection request: "{request.object_description}" '
                f'(image: {cv_image.shape[1]}x{cv_image.shape[0]})'
            )

            # Process detection
            result = self._process_detection(cv_image, request)

            # Fill response
            response.detected = result['detected']
            response.confidence = result['confidence']
            response.center_x = result['center_x']
            response.center_y = result['center_y']
            response.bbox = result['bbox']
            response.phrase = result['phrase']
            response.error_message = result.get('error', '')

            # Add mask to response
            if 'mask' in result and result['mask'] is not None:
                response.mask = result['mask'].flatten().tolist()
                response.mask_height = result['mask'].shape[0]
                response.mask_width = result['mask'].shape[1]
            else:
                response.mask = []
                response.mask_height = 0
                response.mask_width = 0

            elapsed = (time.time() - start_time) * 1000

            if result['detected']:
                self.get_logger().info(
                    f'Detected "{result["phrase"]}" at '
                    f'({result["center_x"]:.1f}, {result["center_y"]:.1f}) '
                    f'with {result["confidence"]:.2f} confidence '
                    f'({elapsed:.1f}ms)'
                )
            else:
                self.get_logger().warn(f'Object not detected: {response.error_message} ({elapsed:.1f}ms)')

            # Publish debug visualization
            if self.publish_debug_images:
                all_detections = result.get('all_detections', [])
                self._publish_debug_image(cv_image, result, all_detections)

            return response

        except Exception as e:
            self.get_logger().error(f'Detection failed: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())
            return self._create_error_response(response, str(e))

    def _process_detection(self, image, request):
        """Process detection with Florence-2"""
        if self.use_mock:
            return self._mock_detect(image, request.object_description)
        else:
            return self._detect_object(image, request.object_description)

    def _detect_object(self, image, text_prompt):
        """Run Florence-2 open vocabulary detection to find the object matching text description"""
        try:
            # Convert to PIL Image
            pil_image = PILImage.fromarray(image)

            # Use Florence-2's Open Vocabulary Detection task
            # This translates to: "Locate {text_prompt} in the image."
            task_prompt = '<OPEN_VOCABULARY_DETECTION>'
            
            # Clean up the prompt - replace underscores with spaces for natural language
            clean_prompt = text_prompt.replace('_', ' ')
            prompt = task_prompt + clean_prompt

            self.get_logger().info(f'Using Florence-2 OVD: "Locate {clean_prompt} in the image."')
            
            # Prepare inputs
            inputs = self.florence2_processor(
                text=prompt,
                images=pil_image,
                return_tensors="pt"
            ).to(self.device, self.florence2_model.dtype)

            # Generate detection
            with torch.no_grad():
                generated_ids = self.florence2_model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=256,
                    num_beams=1,
                    do_sample=False,
                    use_cache=False,
                )

            # Decode results
            generated_text = self.florence2_processor.batch_decode(
                generated_ids,
                skip_special_tokens=False
            )[0]

            self.get_logger().info(f'Florence-2 generated text: {generated_text}')

            # Parse Florence-2 output
            parsed_answer = self.florence2_processor.post_process_generation(
                generated_text,
                task=task_prompt,
                image_size=(pil_image.width, pil_image.height)
            )

            self.get_logger().info(f'Florence-2 parsed output: {parsed_answer}')

            # Extract detection from OVD result
            if task_prompt not in parsed_answer:
                return self._create_detection_result(
                    detected=False,
                    error=f'No object found matching "{text_prompt}"'
                )

            detection_data = parsed_answer[task_prompt]
            bboxes = detection_data.get('bboxes', [])

            if len(bboxes) == 0:
                return self._create_detection_result(
                    detected=False,
                    error=f'No object found matching "{text_prompt}"'
                )

            # Take the first bounding box (OVD should return the most relevant match first)
            bbox = bboxes[0]
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            self.get_logger().info(f'Found "{text_prompt}" at ({cx:.1f}, {cy:.1f}), bbox: {bbox}')

            return self._create_detection_result(
                detected=True,
                confidence=1.0,
                center_x=float(cx),
                center_y=float(cy),
                bbox=[float(x1), float(y1), float(x2), float(y2)],
                phrase=text_prompt,  # Use the original query, not model's label
                mask=None,
                all_detections=[{
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'center_x': float(cx),
                    'center_y': float(cy),
                    'confidence': 1.0,
                    'phrase': text_prompt
                }]
            )

        except Exception as e:
            self.get_logger().error(f'Florence-2 inference failed: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())
            return self._create_detection_result(detected=False, error=str(e))

    def _mock_detect(self, image, text_prompt):
        """Mock detection for testing"""
        h, w = image.shape[:2]
        center_x = w / 2.0
        center_y = h / 2.0
        bbox_width = w * 0.2
        bbox_height = h * 0.2

        x1 = center_x - bbox_width / 2
        y1 = center_y - bbox_height / 2
        x2 = center_x + bbox_width / 2
        y2 = center_y + bbox_height / 2

        self.get_logger().info(f'MOCK: Detected "{text_prompt}" at center ({center_x:.1f}, {center_y:.1f})')

        return self._create_detection_result(
            detected=True,
            confidence=0.95,
            center_x=center_x,
            center_y=center_y,
            bbox=[x1, y1, x2, y2],
            phrase=text_prompt
        )


    def _create_detection_result(self, detected=False, confidence=0.0, center_x=-1.0,
                                  center_y=-1.0, bbox=None, phrase='', error='', mask=None,
                                  all_detections=None):
        """Create a standardized detection result dictionary"""
        return {
            'detected': detected,
            'confidence': confidence,
            'center_x': float(center_x),
            'center_y': float(center_y),
            'bbox': [float(v) for v in bbox] if bbox else [],
            'phrase': phrase,
            'error': error,
            'mask': mask,
            'all_detections': all_detections if all_detections else []
        }

    def _create_error_response(self, response, error_message):
        """Create an error response"""
        response.detected = False
        response.confidence = 0.0
        response.center_x = -1.0
        response.center_y = -1.0
        response.bbox = []
        response.phrase = ''
        response.error_message = error_message
        response.mask = []
        response.mask_height = 0
        response.mask_width = 0
        return response

    def _publish_debug_image(self, image, result, all_detections):
        """Publish debug visualization with bounding boxes"""
        try:
            debug_img = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)

            # Draw all detections with bounding boxes
            for i, det in enumerate(all_detections[:5]):
                color = (0, 255, 0) if i == 0 else (0, 165, 255)  # Green for selected, orange for others

                # Draw bounding box
                x1, y1, x2, y2 = [int(v) for v in det['bbox']]
                cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, 2)

                # Draw center point
                cx, cy = int(det['center_x']), int(det['center_y'])
                cv2.circle(debug_img, (cx, cy), 6, color, -1)
                cv2.circle(debug_img, (cx, cy), 6, (255, 255, 255), 2)

                # Label
                label = f"#{i+1}: {det['phrase']}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(debug_img,
                            (x1, y1 - label_size[1] - 15),
                            (x1 + label_size[0] + 10, y1),
                            color, -1)
                cv2.putText(debug_img, label, (x1 + 5, y1 - 7),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Publish
            debug_img_rgb = cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)
            debug_msg = self.bridge.cv2_to_imgmsg(debug_img_rgb, encoding='rgb8')
            debug_msg.header.stamp = self.get_clock().now().to_msg()
            debug_msg.header.frame_id = 'camera_rgb_optical_frame'
            self.debug_image_pub.publish(debug_msg)
            
        except Exception as e:
            self.get_logger().error(f'Failed to publish debug image: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = Florence2Service()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
