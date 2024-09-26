from pathlib import Path
import pyransac3d as pyrsc
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd


if __name__=='__main__':

    true_length = 40 / 1e3 # m

    pcls = []
    test_folder = Path(__file__).parent.parent.absolute() / "data/length_test"

    for path in sorted(test_folder.iterdir()):
        if 'pcl' in path.name:
            pcls.append(np.loadtxt(test_folder/path))

    pcl = pcls[0]

    
    block_line = pcl[np.where(pcl[:,2] < 1.7e-1)]
    background = pcl[np.where(pcl[:,2] >= 1.7e-1)]

    plt.scatter(pcl[:,0], pcl[:,2], alpha=0.5, s=0.4, label='full profile')
    plt.scatter(block_line[:,0], block_line[:,2], color='r', alpha=0.5, s=0.5, label='block points')

    left = np.argmin(block_line[:,0])
    right = np.argmax(block_line[:,0])

    plt.scatter(block_line[left][0], block_line[left][2])
    plt.scatter(block_line[right][0], block_line[right][2])
       
    approx_length = np.linalg.norm(block_line[left] - block_line[right])
    print(f'approx_length {approx_length * 1000} mm')


    coeffs = np.polyfit(block_line[:,0], block_line[:,2], deg=1)

    left_fit =  np.array([block_line[left][0], coeffs[1] + block_line[left][0] * coeffs[0]])
    right_fit = np.array([block_line[right][0], coeffs[1] + block_line[right][0] * coeffs[0]])

    fit_length = np.linalg.norm(left_fit - right_fit)
    plt.plot([left_fit[0], right_fit[0]], [left_fit[1], right_fit[1]], label='polyfit', linestyle='dashed')
    print(f'Length line poly: {fit_length * 1000} mm')

    print(f"Number of points on the block: {block_line.shape[0]}  -- approx resolution: {1000 * true_length/block_line.shape[0]} mm")

    plt.legend()
    plt.show()