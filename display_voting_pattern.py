#!/usr/bin/env python
# coding: utf-8

# In[26]:


import csv
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D

import sys
import math

if len(sys.argv) == 2:
    congress = sys.argv[1]
else:
    congress = 'H115'

data_folder = 'data/'+congress+'/'
output_folder = 'output/raw/'+congress+'/'+congress+'_'
im_output_folder = 'output/voting_pattern/'+congress+'_vp'
hd_file = 'output/house_details.csv'
trunc = 2

member_count = -1
member_icpsr = {}
with open(data_folder+congress+'_members.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        if not (member_count == -1):
            member_icpsr[row[2]] = member_count
        member_count = member_count + 1
bill_count = -1
bill_roll = {}
bill_title = {}
bill_result = {}
with open(data_folder+congress+'_rollcalls.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        if not bill_count == -1:
            bill_roll[row[2]] = bill_count
            bill_title[bill_count] = row[15]
            if row[14] == "Passed":
                bill_result[bill_count] = 1
            elif row[14] == "Failed":
                bill_result[bill_count] = -1
            else:
                bill_result[bill_count] = 0
        bill_count = bill_count + 1


print("Number of members : " + str(member_count))
print("Number of bills : " + str(bill_count))

A = np.zeros((member_count, bill_count))
I = np.eye(bill_count)

vote_count = -1
abstention_count = 0
with open(data_folder+congress+'_votes.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        if not vote_count == -1:
            if int(row[4]) == 1:
                A[member_icpsr[row[3]]][bill_roll[row[2]]] = 1
            elif int(row[4]) == 6:
                A[member_icpsr[row[3]]][bill_roll[row[2]]] = -1
            else:
                A[member_icpsr[row[3]]][bill_roll[row[2]]] = 0
                if int(row[4]) == 9:
                    abstention_count = abstention_count + 1
        vote_count = vote_count+1

u, s, vh = np.linalg.svd(A)
print(A.shape)

#plt.scatter(vh[0,:], vh[1,:])
#plt.scatter(vh[0,passed_indices], vh[1,passed_indices], c='g')
#plt.scatter(vh[0,failed_indices], vh[1,failed_indices], c='r')
#plt.title('Eigenbills')
#plt.show()
#plt.scatter(u[:n_p,0], u[:n_p,1], c='r')
#plt.scatter(u[n_p+1:,0], u[n_p+1:,1], c='b')
#plt.scatter(u[:,0], u[:,1])
#plt.title('Eigenmembers')


# In[27]:


def std_vot_mat(a):
    a_sorted = np.copy(a)
    [u,s,v]=np.linalg.svd(a);
    a1 = 1;
    a2 = 0;
    b1 = 1;
    b2 = 0;
    mem_inds = sorted(range(u.shape[0]), key=lambda k: b1*u[k,0]+b2*u[k,1])
    bill_inds = sorted(range(v.shape[0]), key=lambda k: a1*v[0,k]+a2*v[1,k])
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            a_sorted[i,j] = a[mem_inds[i], bill_inds[j]]
    return a_sorted, mem_inds, bill_inds

print(A.shape)
a_sorted, _, bill_inds = std_vot_mat(A)
print(len(bill_inds))
[u,s,v]=np.linalg.svd(a_sorted);


# In[28]:


import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
'''
get_ipython().run_line_magic('matplotlib', 'inline')
mem_inds = sorted(range(len(mem)), key=lambda k: mem[k])
bill_inds = sorted(range(len(bill_params)), key=lambda k: 10*np.exp(bill_params[k][0])+0.1*bill_params[k][1])
a_sorted = np.copy(a)
for i in range(a.shape[0]):
    for j in range(a.shape[1]):
        a_sorted[i,j] = a[mem_inds[i], bill_inds[j]]
plt.imshow(a_sorted)
plt.show()
'''
#plt.imshow(A)
#plt.show()
plt.imshow(a_sorted)
figure(figsize=(8,6), dpi=80)
#plt.xlabel('Bills')
#plt.ylabel('Members')
plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
#plt.savefig(im_output_folder)
#plt.show()

import csv
filename = congress + "_sortedbills.csv"
with open(filename, 'w') as csvfile:
    csvwrite = csv.writer(csvfile)
    for x in range(len(bill_inds)):
        #print(str(x+1)+'. '+ bill_title[bill_inds[x]] + '\n')
        csvwrite.writerow([bill_title[bill_inds[x]],bill_result[bill_inds[x]], v[0,x]])
