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
output_folder = 'output/tsnebills/'+congress
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
with open(data_folder+congress+'_rollcalls.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        if not bill_count == -1:
            bill_roll[row[2]] = bill_count
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
result = ['Unkown' for i in range(0, bill_count)]
info = ['-' for i in range(0, bill_count)]
passed_indices = []
failed_indices = []
undecided_indices = []
with open(data_folder+congress+'_rollcalls.csv') as csv_file:
    i = -1
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        if not (i == -1):
            if row[14] == 'Passed':
                passed_indices.append(i)
            elif row[14] == 'Failed':
                failed_indices.append(i)
            else:
                undecided_indices.append(i)
            result[i] = row[14]
            info[i] = row[15] + ", " + row[16]
        i = i + 1

print("Number of Passed bills : " + str(len(passed_indices)))
print("Number of Failed bills : " + str(len(failed_indices)))
print("Number of Undecided bills : " + str(len(undecided_indices)))


from scipy import spatial
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt


print(A.T.shape)
embedding = TSNE(metric='cosine')
embed_vectors = embedding.fit_transform(A.T)
print(embed_vectors.shape)

#fig, ax = plt.subplots()
plt.subplot(1,2,1)
plt.scatter(embed_vectors[passed_indices,0], embed_vectors[passed_indices,1], c='g')
plt.scatter(embed_vectors[failed_indices,0], embed_vectors[failed_indices,1], c='r')
plt.scatter(embed_vectors[undecided_indices,0], embed_vectors[undecided_indices,1], c='y')
plt.title("TSNE embedding of bills")

plt.subplot(1,2,2)
plt.plot([vh[0,i] for i in undecided_indices], [vh[1,i] for i in undecided_indices], 'yo')
plt.plot([vh[0,i] for i in passed_indices], [vh[1,i] for i in passed_indices], 'go')
plt.plot([vh[0,i] for i in failed_indices], [vh[1,i] for i in failed_indices], 'ro')
plt.title("EigVec embedding of bills")
plt.savefig(output_folder)
#plt.show()
#plt.close(fig)
