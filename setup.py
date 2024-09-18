from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'lls_calibration'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ros',
    maintainer_email='luka-groot@hotmail.nl',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            f'lls_calibration = {package_name}.record_calibration_data:main',
            f'calibrate_lls = {package_name}.calibrate_lls:main',
            f'radius_detector = {package_name}.detect_sphere_diameter:main',
        ],
    },
)
