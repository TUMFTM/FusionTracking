"""Setup ros2 package."""
import os
from glob import glob
from setuptools import setup

package_name = "tracking"
submodules = "mypackage/submodules"

setup(
    name=package_name,
    version="0.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name), glob("launch/*.launch.py")),
        (os.path.join("lib", package_name, "config"), glob("tracking/config/*.ini")),
        (
            os.path.join("lib", package_name, "data", "map_data"),
            glob("tracking/data/map_data/*"),
        ),
        (os.path.join("lib", package_name, "src"), glob("tracking/src/*.py")),
        (os.path.join("lib", package_name, "utils"), glob("tracking/utils/*.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    author="Phillip Karle",
    author_email="phillip.karle@tum.de",
    maintainer="Phillip Karle",
    maintainer_email="phillip.karle@tum.de",
    description="Multi Modal Object Fusion and Tracking",
    license="Apache License 2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": ["tracking_node = tracking.tracking_node:main"],
    },
)
