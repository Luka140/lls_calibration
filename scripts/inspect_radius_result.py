from pathlib import Path
import pyransac3d as pyrsc
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd


if __name__=='__main__':

    pcls, tfs = [], []
    test_folder = Path(__file__).parent.parent.absolute() / "data/radius_measurement_2024-09-26T10:05:19.csv"

    # for path in sorted(test_folder.iterdir()):
    #     if 'pcl' in path.name:
    #         pcls.append(np.loadtxt(test_folder/path))
    pcl = pd.read_csv(test_folder/'max_rad_profile.csv')
    true_radius = 0.0725
    radii = []
    centerpoints = []
    runs = 10
    profile = pcl.to_numpy()
    for _ in range(runs):
        center, axis, radius, inlier_idx = pyrsc.Circle().fit(profile, thresh=1e-4)
        radii.append(radius)
        centerpoints.append(center.flatten()[[0,2]])

    rad, centerpts = np.array(radii), np.stack(centerpoints)
    mean_rad, mean_ctpt = np.mean(rad), np.mean(centerpts, axis=0)
    print(mean_rad)
    zero_centered_profile = profile - np.array([mean_ctpt[0], 0, mean_ctpt[1]])
    shifted_profile = zero_centered_profile + np.array([0, 0, mean_rad - true_radius])

    fig, ax = plt.subplots()
    ax.scatter(zero_centered_profile[:,0], zero_centered_profile[:,2], s=1, color='r', label='Profile points')
    ax.scatter(shifted_profile[:,0], shifted_profile[:,2], s=1, color='orange', label='Shifted profile points')


    circle = plt.Circle((0,0), mean_rad, color='b', fill=False, alpha=0.5, label='Predicted circle')
    ax.add_patch(circle)

    circle2 = plt.Circle((0,0), true_radius, color='g', fill=False, alpha=0.5, label='Sphere size')
    ax.add_patch(circle2)

    

    ax.set_aspect('equal', 'box')
    plt.xlabel('x [m]')
    plt.ylabel('z [m]')
    plt.legend()
    plt.show()
