import numpy as np 
from scipy.optimize import least_squares
from pathlib import Path
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt 

"""
All transforms and pointclouds should be repetitions of the same measurement.
The robot is moved around in between the measurements to check the repeatability of the measurements.
For this test, the measurent surface is flat, so the points should simply be a straight line in 3d space.
As the LLS only provides X and Z coordinates, this is simply a linear linefit of a * x + b = z 
"""


def generate_test_data(a, b, dx=0.1):
    x = np.array(np.linspace(0, dx, 10))
    z = a * x + b 
    return np.expand_dims(x, 1), np.expand_dims(z, 1)

# def obj(x, x_vals, z_vals):
#     c1, c2 = x
#     return x_vals * c1 + c2 - z_vals

def obj(x, x_vals, z_vals, poly_order):
    
    return sum([x[i] * x_vals ** i for i in range(poly_order)]) - z_vals

def linefit(pcls, poly_order=10):
    model_pcl = pcls[0]

    x0 = [0] * poly_order
    result = least_squares(obj, x0, args=(model_pcl[:,0], model_pcl[:,2], poly_order))
    # print(result)
    coefs = result.x

    max_residuals = []
    mean_residuals = []

    for pcl in pcls:
        residuals = abs(obj(coefs, pcl[:,0], pcl[:,2], poly_order))
        maximum_deviation = np.max(residuals)
        average_deviation = np.mean(residuals)

        max_residuals.append(maximum_deviation)
        mean_residuals.append(average_deviation)

    return np.array(mean_residuals), np.array(max_residuals)


def test_linefit():
    scale = 100
    a, b = np.random.rand(2) * scale - scale / 2
    xval, zval = generate_test_data(a, b)
    pcl = np.hstack((xval, np.zeros_like(xval), zval))
    pcls = [pcl for _ in range(10)]
    mean_res, max_res = linefit(pcls)
    np.testing.assert_almost_equal(np.zeros_like(max_res), max_res)

if __name__=='__main__':
    test_linefit()

    pcls, tfs = [], []
    test_folder = Path(__file__).parent.parent.absolute() / "data/repeatability_line_set3"

    cols = sorted(list(mcolors.XKCD_COLORS.keys()))[::3]
    for path in sorted(test_folder.iterdir()):
        if 'pcl' in path.name:
            pcl_array = np.loadtxt(test_folder/path)
            pcls.append(pcl_array)
            plt.plot(pcl_array[:,0] * 1000, pcl_array[:,2] * 1000, linewidth=1, label=len(pcls), color=cols[len(pcls)])

    plt.xlabel('x (mm)')
    plt.ylabel('z (mm)')
    plt.legend()
    plt.show()
    mean_residuals, max_residuals = linefit(pcls, 6)

    plt.plot(range(mean_residuals.shape[0]), 1000 * mean_residuals)
    plt.ylabel('Mean residual (mm)')
    plt.xlabel('Recording nr.')
    plt.show()
    print(f"Maximum residuals: {max_residuals * 1000} mm")
    print(f"Average residuals: {mean_residuals * 1000} mm")

