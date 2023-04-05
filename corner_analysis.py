import csv
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D

import sys
import math

if len(sys.argv) >= 2:
    congress = sys.argv[1]
else:
    congress = 'H100'

r_list = [0.05, 0.1, 0.15, 0.2, 0.25]
for r in r_list:
    f_ll_list = []
    f_lr_list = []
    f_ul_list = []
    f_ur_list = []
    c_list = []

    for i in range(95, 117):
        if i <= 99:
            congress = 'H0'+str(i)
        else:
            congress = 'H'+str(i)

        data_folder = 'output/raw/'+congress+'/'+congress+'_'
        output_folder = 'output/raw/'+congress+'/'+congress+'_'
        write_folder = 'output/corner_analysis/'
        hd_file = 'output/house_details.csv'

        bill_mat = np.genfromtxt(output_folder+'eigenbills_squared_normalized.csv', delimiter=',')
        #print(bill_mat.shape)
        bill_mat = bill_mat[:,2:]
        print(bill_mat.shape)

        x_min = np.min(bill_mat[:,0])
        x_max = np.max(bill_mat[:,0])
        y_min = np.min(bill_mat[:,1])
        y_max = np.max(bill_mat[:,1])

#mask = (bill_mat[:, 0] <= x_min+r*(x_max-x_min) and bill_mat[:, 1] <= y_min+r*(y_max-y_min))
        mask1 = (bill_mat[:, 0] <= x_min+r*(x_max-x_min))
        mask2 = (bill_mat[:, 1] <= y_min+r*(y_max-y_min))
#print(mask1*mask2)
#print(bill_mat[mask1*mask2,:].shape)
        f_ll = float(bill_mat[mask1*mask2,:].shape[0]/bill_mat.shape[0])

        mask1 = (bill_mat[:, 0] >= x_max-r*(x_max-x_min))
        mask2 = (bill_mat[:, 1] <= y_min+r*(y_max-y_min))
#print(mask1*mask2)
#print(bill_mat[mask1*mask2,:].shape)
        f_lr = float(bill_mat[mask1*mask2,:].shape[0]/bill_mat.shape[0])

        mask1 = (bill_mat[:, 0] <= x_min+r*(x_max-x_min))
        mask2 = (bill_mat[:, 1] >= y_max-r*(y_max-y_min))
#print(mask1*mask2)
#print(bill_mat[mask1*mask2,:].shape)
        f_ul = float(bill_mat[mask1*mask2,:].shape[0]/bill_mat.shape[0])

        mask1 = (bill_mat[:, 0] >= x_max-r*(x_max-x_min))
        mask2 = (bill_mat[:, 1] >= y_max-r*(y_max-y_min))
#print(mask1*mask2)
#print(bill_mat[mask1*mask2,:].shape)
        f_ur = float(bill_mat[mask1*mask2,:].shape[0]/bill_mat.shape[0])

        print([f_ll, f_lr, f_ul, f_ur])
        f_ll_list.append(f_ll)
        f_lr_list.append(f_lr)
        f_ul_list.append(f_ul)
        f_ur_list.append(f_ur)
        c_list.append(f_ll + f_lr + f_ul + f_ur)

    plt.title(str(int(100*r)) + '% corner bills')
    plt.ylabel('Fraction of corner bills')
    plt.xlabel('Congress index')
    plt.plot([i for i in range(95,117)], f_ll_list, label='lower left')
    plt.plot([i for i in range(95,117)], f_lr_list, label='lower right')
    plt.plot([i for i in range(95,117)], f_ul_list, label='upper left')
    plt.plot([i for i in range(95,117)], f_ur_list, label='upper right')
    plt.plot([i for i in range(95,117)], c_list, label = 'all corners')
    plt.legend()
    plt.savefig(write_folder+'corner_bills_'+str(int(100*r)))
    plt.show()
