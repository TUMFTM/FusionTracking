"""Main functionalities for multi-modal sensor fusion and object tracking."""
import signal
import time

import rclpy

from utils.ros2_interface import ROS2Handler
from utils.setup_helpers import create_path_dict, stamp2time

from src.object_handler import ObjectHandler
from src.physics_prediction import PhysicsPrediction
from src.track_boundary import TrackBoundary

PATH_DICT = create_path_dict()


class TrackingNode(ROS2Handler):
    """Class object central for multi-modal object fusion and tracking."""

    def __init__(self):
        """Create logger, get all params, initialize ROS2 node and parameters."""
        # Initialize ros2 handler
        super().__init__(path_dict=PATH_DICT)

        # Get map data
        self.track_boundary = TrackBoundary(
            all_params=self.params,
        )
        self.pit_boundary = TrackBoundary(
            all_params=self.params,
            track_path=self.params["pit_path"],
        )

        # Initialize object handler
        self.object_handler = ObjectHandler(
            all_params=self.params,
            track_boundary_cls=self.track_boundary,
            pit_boundary_cls=self.pit_boundary,
            get_clock=self.get_clock,
        )

        # Initialize physics-based prediction
        self.physics_prediction = PhysicsPrediction(
            all_params=self.params,
            track_boundary_cls=self.track_boundary,
            pit_boundary_cls=self.pit_boundary,
        )

    def timer_callback(self):
        """Step function as ROS 2 callback."""
        # Start time measurement
        start_time = time.time_ns()

        # Set module state
        self.module_state = 30

        # Update time stamp
        self.object_handler.sync_time()

        # Process received detections
        self.module_state = self.object_handler.process_detections(
            detections_subscriber_dict=self.detection_subscriber_dict,
        )

        # Log latency
        self.latency_dict["t_detect"] = stamp2time(
            *self.get_clock().now().seconds_nanoseconds()
        )

        # Check if ego status is valid
        self.check_ego_status()

        # Conduct tracking step
        self.object_handler(
            detections_subscriber_dict=self.detection_subscriber_dict,
            module_state=self.module_state,
        )

        # Log latency
        self.latency_dict["t_track"] = stamp2time(
            *self.get_clock().now().seconds_nanoseconds()
        )

        # Physics-based prediction
        self.physics_prediction(
            obs_storage=self.object_handler.observation_storage,
            obj_filters=self.object_handler.obj_filters,
            ego_state=self.object_handler.ego_state,
        )

        # Log latency
        self.latency_dict["t_pred"] = stamp2time(
            *self.get_clock().now().seconds_nanoseconds()
        )

        # Check module state
        self.check_module_state()

        # Publish topics
        self.publish()

        # Log latency
        self.latency_dict["t_send"] = stamp2time(
            *self.get_clock().now().seconds_nanoseconds()
        )

        # Log data
        self.log_data(start_time=start_time)

    def log_data(self, start_time: int):
        """Log tracking data.

        Args:
            start_time (int): ros2-time in ns at cycle start.
        """
        # Log tracking data
        if (
            self.object_handler.bool_detection_input
            or self.object_handler.ego_state["is_updated"]
        ):
            self.data_logger.log_data(
                time_ns=stamp2time(*self.get_clock().now().seconds_nanoseconds()),
                detection_input=self.object_handler.detection_input,
                ego_state=self.object_handler.ego_state,
                tracking_input=self.object_handler.preprocessed_tracking_input_list,
                match_dict=self.object_handler.matching.match_dict,
                filter_log=self.object_handler.filter_log,
                object_dict=self.object_handler.estimated_object_dict,
                pred_dict=self.physics_prediction.pred_dict,
                cycle_time_ns=time.time_ns() - start_time,
            )

        # Log latency data
        self.latency_logger.log_latency(
            latency_dict=self.latency_dict,
            detection_subscriber_dict=self.detection_subscriber_dict,
            len_obsv_storage=len(self.object_handler.observation_storage),
            pred_dict=self.physics_prediction.pred_dict,
        )

        # report time violation
        if self.object_handler.is_time_violation:
            self.latency_logger.output_latency_data(
                msg_logger=self.msg_logger.__getattribute__("warning"),
            )
            self.object_handler.is_time_violation = False


def main(args=None):
    """Execute tracking_node."""
    rclpy.init(args=args)

    tracking_node = TrackingNode()

    rclpy.spin(tracking_node)

    signal.signal(signal.SIGINT, tracking_node.shutdown_ros)

    rclpy.shutdown()

    tracking_node.destroy_node()


if __name__ == "__main__":
    main()
