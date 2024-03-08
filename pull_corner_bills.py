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
f_ll_list = []
f_lr_list = []
f_ul_list = []
f_ur_list = []
c_list = []

r = 0.1

for i in range(95, 116):
    print('Looking at congress '+str(i))
    if i <= 99:
        congress = 'H0'+str(i)
    else:
        congress = 'H'+str(i)

    data_folder = 'output/raw/'+congress+'/'+congress+'_'
    output_folder = 'output/raw/'+congress+'/'+congress+'_'
    write_folder = 'output/corner_analysis/'
    hd_file = 'output/house_details.csv'
    file_name = congress+'_corner_bills'

    bill_mat = np.genfromtxt(output_folder+'eigenbills_squared_normalized.csv', delimiter=',')
#print(bill_mat.shape)
    bill_mat = bill_mat[:,2:]
    print(bill_mat.shape)

    bill_details = []
    with open(output_folder+'eigenbills.csv') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        for row in csv_reader:
            bill_details.append(row)

    x_min = np.min(bill_mat[:,0])
    x_max = np.max(bill_mat[:,0])
    y_min = np.min(bill_mat[:,1])
    y_max = np.max(bill_mat[:,1])

#mask = (bill_mat[:, 0] <= x_min+r*(x_max-x_min) and bill_mat[:, 1] <= y_min+r*(y_max-y_min))
    mask1 = (bill_mat[:, 0] <= x_min+r*(x_max-x_min))
    mask2 = (bill_mat[:, 1] <= y_min+r*(y_max-y_min))
#print(mask1*mask2)
#print(bill_mat[mask1*mask2,:].shape)
    f_ll = []
    mask = mask1*mask2
    for i in range(mask.shape[0]):
        if mask[i] == True:
            f_ll.append(i)
    print(f_ll)

    mask1 = (bill_mat[:, 0] >= x_max-r*(x_max-x_min))
    mask2 = (bill_mat[:, 1] <= y_min+r*(y_max-y_min))
    print(mask1*mask2)
#print(mask1*mask2)
#print(bill_mat[mask1*mask2,:].shape)
    f_lr = []
    mask = mask1*mask2
    for i in range(mask.shape[0]):
        if mask[i] == True:
            f_lr.append(i)
    print(f_lr)

    mask1 = (bill_mat[:, 0] <= x_min+r*(x_max-x_min))
    mask2 = (bill_mat[:, 1] >= y_max-r*(y_max-y_min))
#print(mask1*mask2)
#print(bill_mat[mask1*mask2,:].shape)
    f_ul = []
    mask = mask1*mask2
    for i in range(mask.shape[0]):
        if mask[i] == True:
            f_ul.append(i)
    print(f_ul)

    mask1 = (bill_mat[:, 0] >= x_max-r*(x_max-x_min))
    mask2 = (bill_mat[:, 1] >= y_max-r*(y_max-y_min))
#print(mask1*mask2)
#print(bill_mat[mask1*mask2,:].shape)
    f_ur = []
    mask = mask1*mask2
    for i in range(mask.shape[0]):
        if mask[i] == True:
            f_ur.append(i)
    print(f_ur)

    with open(write_folder+file_name+'.csv','w') as csvfile:
        writercsv = csv.writer(csvfile)
        myCsvRow = ['\n', '>>>>>', 'Lower', 'Left', '<<<<<', '\n']
        writercsv.writerow(myCsvRow)
        for i in f_ll:
            myCsvRow = bill_details[i]
            writercsv.writerow(myCsvRow)
        myCsvRow = ['\n', '>>>>>', 'Lower', 'Right', '<<<<<', '\n']
        writercsv.writerow(myCsvRow)
        for i in f_lr:
            myCsvRow = bill_details[i]
            writercsv.writerow(myCsvRow)
        myCsvRow = ['\n', '>>>>>', 'Upper', 'Left', '<<<<<', '\n']
        writercsv.writerow(myCsvRow)
        for i in f_ul:
            myCsvRow = bill_details[i]
            writercsv.writerow(myCsvRow)
        myCsvRow = ['\n', '>>>>>', 'Upper', 'Right', '<<<<<', '\n']
        writercsv.writerow(myCsvRow)
        for i in f_ur:
            myCsvRow = bill_details[i]
            writercsv.writerow(myCsvRow)
