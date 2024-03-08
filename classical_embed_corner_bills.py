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

r = 0.2

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
                if token.lower() not in tokens:
                    tokens[token.lower()] = 1
                else:
                    tokens[token.lower()] = tokens[token.lower()]+1

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

import operator
sorted_tokens = sorted(tokens.items(), key=operator.itemgetter(1), reverse=True)

import copy
for i in range(1):
    stopwords = [word[0] for word in sorted_tokens[:i]]
    print(stopwords)

    f_ll_b = copy.deepcopy(f_ll_bills)
    f_lr_b = copy.deepcopy(f_lr_bills)
    f_ul_b = copy.deepcopy(f_ul_bills)
    f_ur_b = copy.deepcopy(f_ur_bills)
    f_b = copy.deepcopy(f_bills)

    all_b = f_ll_b+f_lr_b+f_ul_b+f_ur_b+f_b
    for i in range(len(all_b)):
        #print(all_b[i])
        querywords = all_b[i].split()
        resultwords  = [word for word in querywords if word.lower() not in stopwords]
        result = ' '.join(resultwords)
        all_b[i] = result
        #print(all_b[i])
    
    vectorized = vectorize(all_b)
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

    from scipy import spatial
    from sklearn.manifold import TSNE
    from matplotlib import pyplot as plt
    import matplotlib.patches as mpatches

    vectors_corner = vectors_ll + vectors_lr + vectors_ul + vectors_ur
    vectors_full = vectors_corner + vectors
    vectors_full = np.stack(vectors_full).squeeze()
    print(vectors_full.shape)
    embedding = TSNE(metric='cosine')
    embed_vectors = embedding.fit_transform(vectors_full)
    print(embed_vectors.shape)
    '''
    colors = np.zeros((vectors_full.shape[0],))
    for i in range(vectors_full.shape[0]):
        if i < len(f_ll_bills):
            colors[i] = 0.2
        elif i < len(f_ll_bills)+len(f_lr_bills):
            colors[i] = 0.4
        elif i < len(f_ll_bills)+len(f_lr_bills)+len(f_ul_bills):
            colors[i] = 0.6
        elif i < len(f_ll_bills)+len(f_lr_bills)+len(f_ul_bills)+len(f_ur_bills):
            colors[i] = 0.8
        elif i < len(f_ll_bills)+len(f_lr_bills)+len(f_ul_bills)+len(f_ur_bills)+len(f_bills):
            colors[i] = 1
    
    cmap = plt.get_cmap('viridis')
    print(cmap)
    ll = mpatches.Patch(color=cmap[0.4], label='LL corner')
    lr = mpatches.Patch(color=0.55, label='LR corner')
    ul = mpatches.Patch(color=0.7, label='UL corner')
    ur = mpatches.Patch(color=0.85, label='UR corner')
    nc = mpatches.Patch(color=1, label='non-corner')
    plt.legend(handles=[ll,lr,ul,ur,nc])

    #ax.scatter(embed_vectors[:len(f_ll_bills),0], embed_vectors[:len(f_ll_bills),1], c=colors[:len(f_ll_bills)], label='LL corner')
    #ax.scatter(embed_vectors[len(f_ll_bills):len(f_ll_bills)+len(f_lr_bills),0], embed_vectors[len(f_ll_bills):len(f_ll_bills)+len(f_lr_bills),1], c=colors[len(f_ll_bills):len(f_ll_bills)+len(f_lr_bills)], label='LR corner')
    '''
    
    fig, ax = plt.subplots()
    ax.scatter(embed_vectors[:len(f_ll_bills),0], embed_vectors[:len(f_ll_bills),1], c='r', label='LL corner')
    ax.scatter(embed_vectors[len(f_ll_bills):len(f_ll_bills)+len(f_lr_bills),0], embed_vectors[len(f_ll_bills):len(f_ll_bills)+len(f_lr_bills),1], c='b', label='LR corner')
    ax.scatter(embed_vectors[len(f_ll_bills)+len(f_lr_bills):len(f_ll_bills)+len(f_lr_bills)+len(f_ul_bills),0], embed_vectors[len(f_ll_bills)+len(f_lr_bills):len(f_ll_bills)+len(f_lr_bills)+len(f_ul_bills),1], c='g', label='UL corner')
    ax.scatter(embed_vectors[len(f_ll_bills)+len(f_lr_bills)+len(f_ul_bills):len(f_ll_bills)+len(f_lr_bills)+len(f_ul_bills)+len(f_ur_bills),0], embed_vectors[len(f_ll_bills)+len(f_lr_bills)+len(f_ul_bills):len(f_ll_bills)+len(f_lr_bills)+len(f_ul_bills)+len(f_ur_bills),1], c='y', label='UR corner')
    ax.scatter(embed_vectors[len(f_ll_bills)+len(f_lr_bills)+len(f_ul_bills)+len(f_ur_bills):,0], embed_vectors[len(f_ll_bills)+len(f_lr_bills)+len(f_ul_bills)+len(f_ur_bills):,1], c='k', label='non-corner')
    ax.legend()
    plt.show()

    count = 0
    avg1 = 0.0
    avg = 0.0
    for i in range(len(vectors_ll)):
        for j in range(i+1, len(vectors_ll)):
            count = count + 1
            avg = (count-1)*avg/count + spatial.distance.cosine(vectors_ll[i], vectors_ll[j])/count
    avg1 = avg
    count = 0
    avg2 = 0
    avg = 0
    for i in range(len(vectors)):
        for j in range(len(vectors_ll)):
            count = count + 1
            avg = (count-1)*avg/count + spatial.distance.cosine(vectors[i], vectors_ll[j])/count
    avg2 = avg
    print('Avg dist b/w ll corner bills, Avg dist b/w ll corner and non-corner bills : ' + str(round(avg1,2)) + ',' + str(round(avg2, 2)))
    print('Percentage increase : ' + str(round(((avg2-avg1)/avg1),2)))

    count = 0
    avg1 = 0.0
    avg = 0.0
    for i in range(len(vectors_lr)):
        for j in range(i+1, len(vectors_lr)):
            count = count + 1
            avg = (count-1)*avg/count + spatial.distance.cosine(vectors_lr[i], vectors_lr[j])/count
    avg1 = avg
    count = 0
    avg2 = 0
    avg = 0
    for i in range(len(vectors)):
        for j in range(len(vectors_lr)):
            count = count + 1
            avg = (count-1)*avg/count + spatial.distance.cosine(vectors[i], vectors_lr[j])/count
    avg2 = avg
    print('Avg dist b/w lr corner bills, Avg dist b/w lr corner and non-corner bills : ' + str(round(avg1,2)) + ',' + str(round(avg2, 2)))
    print('Percentage increase : ' + str(round(((avg2-avg1)/avg1),2)))

    count = 0
    avg1 = 0.0
    avg = 0.0
    for i in range(len(vectors_ul)):
        for j in range(i+1, len(vectors_ul)):
            count = count + 1
            avg = (count-1)*avg/count + spatial.distance.cosine(vectors_ul[i], vectors_ul[j])/count
    avg1 = avg
    count = 0
    avg2 = 0
    avg = 0
    for i in range(len(vectors)):
        for j in range(len(vectors_ul)):
            count = count + 1
            avg = (count-1)*avg/count + spatial.distance.cosine(vectors[i], vectors_ul[j])/count
    avg2 = avg
    print('Avg dist b/w ul corner bills, Avg dist b/w ul corner and non-corner bills : ' + str(round(avg1,2)) + ',' + str(round(avg2, 2)))
    print('Percentage increase : ' + str(round(((avg2-avg1)/avg1),2)))

    count = 0
    avg1 = 0.0
    avg = 0.0
    for i in range(len(vectors_ur)):
        for j in range(i+1, len(vectors_ur)):
            count = count + 1
            avg = (count-1)*avg/count + spatial.distance.cosine(vectors_ur[i], vectors_ur[j])/count
    avg1 = avg
    count = 0
    avg2 = 0
    avg = 0
    for i in range(len(vectors)):
        for j in range(len(vectors_ur)):
            count = count + 1
            avg = (count-1)*avg/count + spatial.distance.cosine(vectors[i], vectors_ur[j])/count
    avg2 = avg
    print('Avg dist b/w ur corner bills, Avg dist b/w ur corner and non-corner bills : ' + str(round(avg1,2)) + ',' + str(round(avg2, 2)))
    print('Percentage increase : ' + str(round(((avg2-avg1)/avg1),2)))

