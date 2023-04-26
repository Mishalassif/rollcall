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

f_ll_bills = []
f_lr_bills = []
f_ul_bills = []
f_ur_bills = []
f_bills = []

total = 0
for i in range(115, 116):
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
    #print(bill_mat.shape)

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
    #print(f_ll)

    mask1 = (bill_mat[:, 0] >= x_max-r*(x_max-x_min))
    mask2 = (bill_mat[:, 1] <= y_min+r*(y_max-y_min))
    #print(mask1*mask2)
#print(mask1*mask2)
#print(bill_mat[mask1*mask2,:].shape)
    f_lr = []
    mask = mask1*mask2
    for i in range(mask.shape[0]):
        if mask[i] == True:
            f_lr.append(i)
    #print(f_lr)

    mask1 = (bill_mat[:, 0] <= x_min+r*(x_max-x_min))
    mask2 = (bill_mat[:, 1] >= y_max-r*(y_max-y_min))
#print(mask1*mask2)
#print(bill_mat[mask1*mask2,:].shape)
    f_ul = []
    mask = mask1*mask2
    for i in range(mask.shape[0]):
        if mask[i] == True:
            f_ul.append(i)
    #print(f_ul)

    mask1 = (bill_mat[:, 0] >= x_max-r*(x_max-x_min))
    mask2 = (bill_mat[:, 1] >= y_max-r*(y_max-y_min))
#print(mask1*mask2)
#print(bill_mat[mask1*mask2,:].shape)
    f_ur = []
    mask = mask1*mask2
    for i in range(mask.shape[0]):
        if mask[i] == True:
            f_ur.append(i)
    #print(f_ur)

    for i in f_ll:
        f_ll_bills.append(bill_details[i][-1])
    for i in f_lr:
        f_lr_bills.append(bill_details[i][-1])
    for i in f_ul:
        f_ul_bills.append(bill_details[i][-1])
    for i in f_ur:
        f_ur_bills.append(bill_details[i][-1])
    for i in range(len(bill_details)):
        if i not in f_ll and i not in f_lr and i not in f_ul and i not in f_ur:
            f_bills.append(bill_details[i][-1])

    total = total + len(bill_details)

print('Total bills:')
print(len(f_ll_bills)+len(f_lr_bills)+len(f_ul_bills)+len(f_ur_bills)+len(f_bills))
print(len(f_ll_bills))
print(len(f_lr_bills))
print(len(f_ul_bills))
print(len(f_ur_bills))
print(len(f_bills))
print(total)
from sent2vec.vectorizer import Vectorizer

vectorizer = Vectorizer()
vectorizer.run(f_ll_bills+f_lr_bills+f_ul_bills+f_ur_bills+f_bills)
print('Done Running Vectorization')
if len(f_ll_bills) == 0:
    vectors_ll = []
else:
    #vectorizer = Vectorizer()
    #vectorizer.run(f_ll_bills)
    #vectors_ll = vectorizer.vectors.copy()
    vectors_ll = vectorizer.vectors[0:len(f_ll_bills)]

if len(f_lr_bills) == 0:
    vectors_lr = []
else:
    #vectorizer = Vectorizer()
    #vectorizer.run(f_lr_bills)
    #vectors_ll = vectorizer.vectors.copy()
    vectors_lr = vectorizer.vectors[len(f_ll_bills):len(f_ll_bills)+len(f_lr_bills)]

if len(f_ul_bills) == 0:
    vectors_ul = []
else:
    #vectorizer = Vectorizer()
    #vectorizer.run(f_ul_bills)
    #vectors_ul = vectorizer.vectors.copy()
    vectors_ul = vectorizer.vectors[len(f_ll_bills)+len(f_lr_bills):len(f_ll_bills)+len(f_lr_bills)+len(f_ul_bills)]

if len(f_ur_bills) == 0:
    vectors_ur = []
else:
    #vectorizer = Vectorizer()
    #vectorizer.run(f_ur_bills)
    #vectors_ur = vectorizer.vectors.copy()
    vectors_ur = vectorizer.vectors[len(f_ll_bills)+len(f_lr_bills)+len(f_ul_bills):len(f_ll_bills)+len(f_lr_bills)+len(f_ul_bills)+len(f_ur_bills)]

if len(f_bills) == 0:
    vectors = []
else:
    #vectorizer = Vectorizer()
    #vectorizer.run(f_bills)
    #vectors = vectorizer.vectors.copy()
    vectors = vectorizer.vectors[len(f_ll_bills)+len(f_lr_bills)+len(f_ul_bills)+len(f_ur_bills):len(f_ll_bills)+len(f_lr_bills)+len(f_ul_bills)+len(f_ur_bills)+len(f_bills)]

print(len(vectors_ll))
print(vectors_ll[0].shape)
print(len(vectors_lr))
print(vectors_lr[0].shape)
print(len(vectors_ul))
print(vectors_ul[0].shape)
print(len(vectors_ur))
print(vectors_ur[0].shape)
print(len(vectors))
print(vectors[0].shape)

print('Done Vectorization')
from scipy import spatial

vectors_corner = vectors_ll + vectors_lr + vectors_ul + vectors_ur

count = 0
avg = 0
for i in range(len(vectors_corner)):
    for j in range(i+1, len(vectors_corner)):
        count = count + 1
        avg = (count-1)*avg/count + spatial.distance.cosine(vectors_corner[i], vectors_corner[j])/count
print('Average distance between corner bills: ')
print(avg)
        

count = 0
avg = 0
for i in range(len(vectors)):
    for j in range(i+1, len(vectors)):
        count = count + 1
        avg = (count-1)*avg/count + spatial.distance.cosine(vectors[i], vectors[j])/count
print('Average distance between non-corner bills: ')
print(avg)

count = 0
avg = 0
for i in range(len(vectors)):
    for j in range(len(vectors_corner)):
        count = count + 1
        avg = (count-1)*avg/count + spatial.distance.cosine(vectors[i], vectors_corner[j])/count
print('Average distance between corner and non-corner bills: ')
print(avg)

count = 0
avg = 0
for i in range(len(vectors_ll)):
    for j in range(i+1, len(vectors_ll)):
        count = count + 1
        avg = (count-1)*avg/count + spatial.distance.cosine(vectors_ll[i], vectors_ll[j])/count
print('Average distance between ll corner bills: ')
print(avg)
        
count = 0
avg = 0
for i in range(len(vectors_lr)):
    for j in range(i+1, len(vectors_lr)):
        count = count + 1
        avg = (count-1)*avg/count + spatial.distance.cosine(vectors_lr[i], vectors_lr[j])/count
print('Average distance between lr corner bills: ')
print(avg)
        
count = 0
avg = 0
for i in range(len(vectors_ul)):
    for j in range(i+1, len(vectors_ul)):
        count = count + 1
        avg = (count-1)*avg/count + spatial.distance.cosine(vectors_ul[i], vectors_ul[j])/count
print('Average distance between ul corner bills: ')
print(avg)
        
count = 0
avg = 0
for i in range(len(vectors_ur)):
    for j in range(i+1, len(vectors_ur)):
        count = count + 1
        avg = (count-1)*avg/count + spatial.distance.cosine(vectors_ur[i], vectors_ur[j])/count
print('Average distance between ur corner bills: ')
print(avg)
        
count = 0
avg = 0
for i in range(len(vectors)):
    for j in range(len(vectors_ll)):
        count = count + 1
        avg = (count-1)*avg/count + spatial.distance.cosine(vectors[i], vectors_ll[j])/count
print('Average distance between ll corner and non-corner bills: ')
print(avg)

count = 0
avg = 0
for i in range(len(vectors)):
    for j in range(len(vectors_lr)):
        count = count + 1
        avg = (count-1)*avg/count + spatial.distance.cosine(vectors[i], vectors_lr[j])/count
print('Average distance between lr corner and non-corner bills: ')
print(avg)

count = 0
avg = 0
for i in range(len(vectors)):
    for j in range(len(vectors_ul)):
        count = count + 1
        avg = (count-1)*avg/count + spatial.distance.cosine(vectors[i], vectors_ul[j])/count
print('Average distance between ul corner and non-corner bills: ')
print(avg)

count = 0
avg = 0
for i in range(len(vectors)):
    for j in range(len(vectors_ur)):
        count = count + 1
        avg = (count-1)*avg/count + spatial.distance.cosine(vectors[i], vectors_ur[j])/count
print('Average distance between ur corner and non-corner bills: ')
print(avg)
