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

tokens = dict()
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
    bill_mat = bill_mat[:,2:]

    bill_details = []
    with open(output_folder+'eigenbills.csv') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        for row in csv_reader:
            bill_details.append(row)
            for token in bill_details[-1][-1].split(' '):
                tokens[token.lower()] = 1

    x_min = np.min(bill_mat[:,0])
    x_max = np.max(bill_mat[:,0])
    y_min = np.min(bill_mat[:,1])
    y_max = np.max(bill_mat[:,1])

    mask1 = (bill_mat[:, 0] <= x_min+r*(x_max-x_min))
    mask2 = (bill_mat[:, 1] <= y_min+r*(y_max-y_min))
    
    f_ll = []
    mask = mask1*mask2
    for i in range(mask.shape[0]):
        if mask[i] == True:
            f_ll.append(i)

    mask1 = (bill_mat[:, 0] >= x_max-r*(x_max-x_min))
    mask2 = (bill_mat[:, 1] <= y_min+r*(y_max-y_min))

    f_lr = []
    mask = mask1*mask2
    for i in range(mask.shape[0]):
        if mask[i] == True:
            f_lr.append(i)

    mask1 = (bill_mat[:, 0] <= x_min+r*(x_max-x_min))
    mask2 = (bill_mat[:, 1] >= y_max-r*(y_max-y_min))
    
    f_ul = []
    mask = mask1*mask2
    for i in range(mask.shape[0]):
        if mask[i] == True:
            f_ul.append(i)

    mask1 = (bill_mat[:, 0] >= x_max-r*(x_max-x_min))
    mask2 = (bill_mat[:, 1] >= y_max-r*(y_max-y_min))
    
    f_ur = []
    mask = mask1*mask2
    for i in range(mask.shape[0]):
        if mask[i] == True:
            f_ur.append(i)

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

print('Number of Tokens collected:')
#print(tokens.keys())
print(len(tokens.keys()))

print('Total bills:')
print(len(f_ll_bills)+len(f_lr_bills)+len(f_ul_bills)+len(f_ur_bills)+len(f_bills))
print(len(f_ll_bills))
print(len(f_lr_bills))
print(len(f_ul_bills))
print(len(f_ur_bills))
print(len(f_bills))
print(total)

def vectorize(sent_list):
    vec_list = []
    for sent in sent_list:
        vec = np.zeros((len(tokens.keys()),1))
        vec_dict = dict()
        for token in tokens.keys():
            vec_dict[token] = 0
        for token in sent.split(' '):
            vec_dict[token.lower()] = vec_dict[token.lower()]+1
        i = 0
        for token in tokens.keys():
            vec[i] = vec_dict[token]
            i = i + 1
        vec = vec/sum(vec)
        vec_list.append(vec)
    return vec_list

vectorized = vectorize(f_ll_bills+f_lr_bills+f_ul_bills+f_ur_bills+f_bills)
print('Done Running Vectorization')
if len(f_ll_bills) == 0:
    vectors_ll = []
else:
    vectors_ll = vectorized[0:len(f_ll_bills)]

if len(f_lr_bills) == 0:
    vectors_lr = []
else:
    vectors_lr = vectorized[len(f_ll_bills):len(f_ll_bills)+len(f_lr_bills)]

if len(f_ul_bills) == 0:
    vectors_ul = []
else:
    vectors_ul = vectorized[len(f_ll_bills)+len(f_lr_bills):len(f_ll_bills)+len(f_lr_bills)+len(f_ul_bills)]

if len(f_ur_bills) == 0:
    vectors_ur = []
else:
    vectors_ur = vectorized[len(f_ll_bills)+len(f_lr_bills)+len(f_ul_bills):len(f_ll_bills)+len(f_lr_bills)+len(f_ul_bills)+len(f_ur_bills)]

if len(f_bills) == 0:
    vectors = []
else:
    vectors = vectorized[len(f_ll_bills)+len(f_lr_bills)+len(f_ul_bills)+len(f_ur_bills):len(f_ll_bills)+len(f_lr_bills)+len(f_ul_bills)+len(f_ur_bills)+len(f_bills)]

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
