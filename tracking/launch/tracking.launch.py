"""Launch script of ros2-node."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description of ros2-node."""
    # Initialize launch parameters
    frequency = LaunchConfiguration("frequency", default=50.0)
    max_delay_ego_s = LaunchConfiguration("max_delay_ego_s", default=0.15)
    checks_enabled = LaunchConfiguration("checks_enabled", default=False)
    track = LaunchConfiguration("track", default="LVMS")
    use_sim_time = LaunchConfiguration("use_sim_time", default=False)
    ego_raceline = LaunchConfiguration("ego_raceline", default="default")
    send_prediction = LaunchConfiguration("send_prediction", default=True)

    return LaunchDescription(
        [
            # Declare Arguments
            DeclareLaunchArgument(
                "frequency",
                default_value=frequency,
                description="Specify node frequency in Hz",
            ),
            DeclareLaunchArgument(
                "max_delay_ego_s",
                default_value=max_delay_ego_s,
                description="Specify maximal delay of received ego state in s",
            ),
            DeclareLaunchArgument(
                "checks_enabled",
                default_value=checks_enabled,
                description="Set bool to activate safety checks",
            ),
            DeclareLaunchArgument(
                "track",
                default_value=track,
                description="Specify track to use correct map, 'LVMS' or 'IMS'",
            ),
            DeclareLaunchArgument(
                "use_sim_time",
                default_value=use_sim_time,
                description="Set node to use sim time to replay ros2-bags",
            ),
            DeclareLaunchArgument(
                "ego_raceline",
                default_value=ego_raceline,
                description="Set ego racline: default, inner, outer, center",
            ),
            DeclareLaunchArgument(
                "send_prediction",
                default_value=send_prediction,
                description="Enable prediction publish",
            ),
            # Create Node
            Node(
                package="tracking",
                executable="tracking_node",
                name="TRACKING",
                namespace="",
                parameters=[
                    {
                        "frequency": frequency,
                        "max_delay_ego_s": max_delay_ego_s,
                        "checks_enabled": checks_enabled,
                        "track": track,
                        "use_sim_time": use_sim_time,
                        "ego_raceline": ego_raceline,
                        "send_prediction": send_prediction,
                    }
                ],
                arguments=["--ros-args"],
            ),
        ]
    )
