import numpy as np 
from scipy.optimize import lsq_linear
from pathlib import Path

"""
All transforms and pointclouds should be repetitions of the same measurement.
The robot is moved around in between the measurements to check the repeatability of the measurements.
For this test, the measurent surface is flat, so the points should simply be a straight line in 3d space.
As the LLS only provides X and Z coordinates, this is simply a linear linefit of a * x = z 
"""


def generate_test_data(a, b, dx=0.1):
    x = np.linspace(0, dx, 10)
    z = a * x + b 


pcls, tfs = [], []
test_folder = Path(__file__).parent.parent.absolute() / "data/repeatability_line_set0"

for path in sorted(test_folder.iterdir()):
    if 'pcl' in path.name:
        pcls.append(np.loadtxt(test_folder/path))

model_pcl = pcls.pop(0)

A = np.expand_dims(model_pcl[:,0], axis=1)
b = model_pcl[:,2]

result = lsq_linear(A,b)
coeff = result.x[0]

max_residuals = []
mean_residuals = []

for pcl in pcls:
    residuals = np.abs(pcl[:,0] * coeff - pcl[:,2])
    maximum_deviation = np.max(residuals)
    average_deviation = np.mean(residuals)

    max_residuals.append(maximum_deviation)
    mean_residuals.append(average_deviation)

print(f"Maximum residuals: {max_residuals}")
print(f"Average residuals: {mean_residuals}")


