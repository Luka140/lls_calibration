from pathlib import Path
import pyransac3d as pyrsc
import numpy as np
import matplotlib.pyplot as plt 


def test_consistency_ransac(profile, runs=30, ransac_threshold=1e-4):
    """_
    Calculates the Mean Absolute Deviation (MAD) of the RANSAC profile radius and centerpoint results.
    """
    radii = []
    centerpoints = []

    for _ in range(runs):
        center, axis, radius, inlier_idx = pyrsc.Circle().fit(profile, thresh=ransac_threshold)
        radii.append(radius)
        centerpoints.append(center.flatten()[[0,2]])

    rad, centerpts = np.array(radii), np.stack(centerpoints)
    mean_rad, mean_ctpt = np.mean(rad), np.mean(centerpts, axis=0)
    mad_rad, mad_ctp = np.mean(np.abs(rad-mean_rad)), np.mean(np.abs(centerpts-mean_ctpt), axis=0)
    return mad_rad, mad_ctp


if __name__=='__main__':

    pcls, tfs = [], []
    test_folder = Path(__file__).parent.parent.absolute() / "data/set11"

    for path in sorted(test_folder.iterdir()):
        if 'pcl' in path.name:
            pcls.append(np.loadtxt(test_folder/path))

    # for pcl in pcls:
    #     center, axis, radius, inlier_idx = pyrsc.Circle().fit(pcl, thresh=5e-5)

    #     print(f'Center: {np.array2string(center, suppress_small=True, floatmode="fixed", precision=4)}  -  Radius: {1000 * radius:.3f} mm  -  Inliers: {100 * inlier_idx.shape[0] / pcl.shape[0]:.3f}%')

    # fig = plt.figure()
    # ax = fig.add_subplot()
    # ax.set_aspect('equal', 'box')
    # ax.scatter(pcl[:,0], pcl[:,2], s=0.02, alpha=1)
    # plt.show()

    for i, pcl in enumerate(pcls):
        rad_std, ctp_std = test_consistency_ransac(pcl, runs=30, ransac_threshold=1e-4)
        print(f"profile {i}: radius mean abs deviation: {1000 * rad_std:.4f} mm  -  centerpoint mean abs deviation: {np.array2string(1000 * ctp_std, suppress_small=True, floatmode='fixed', precision=4)} mm")


