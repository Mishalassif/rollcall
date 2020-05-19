import csv
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D

import sys

if len(sys.argv) == 2:
    congress = sys.argv[1]
else:
    congress = 'H115'

data_folder = 'data/'+congress+'/'
output_folder = 'output/'+congress+'/'+congress+'_'

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

print "Number of members : " + str(member_count)
print "Number of bills : " + str(bill_count)

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

print A
print "Number of votes : " + str(vote_count)
print "Number of abstentions : " + str(abstention_count)

yn_vote_count = 0
for i in range(0, member_count):
    for j in range(0, bill_count):
        yn_vote_count = yn_vote_count + abs(A[i][j])

print "Fraction of votes : " + str(float(vote_count)/(member_count*bill_count))
print (yn_vote_count)/vote_count
print (yn_vote_count)/(member_count*bill_count)
print (yn_vote_count+abstention_count)/vote_count
print (yn_vote_count+abstention_count)/(member_count*bill_count)

u, s, vh = np.linalg.svd(A)
smat = np.zeros((member_count, bill_count))
smat[:member_count, :member_count] = np.diag(s)
A_d = (u[:,:trunc].dot(smat[:trunc,:trunc].dot(vh[:trunc,:])))
rel_energy = np.copy(s)
mar_energy = np.copy(s)
total_energy = (np.linalg.norm(A))*(np.linalg.norm(A))

print "Max EVs : " + str(s[0]) + ", " + str(s[1])
print "Approximation error(p) in l2 : " + str(100*np.linalg.norm(A_d-A)/np.linalg.norm(A))

cutoff = 0
for i in range(0, len(s)-1):
    rel_energy[i] = 100*(s[i]*s[i])/total_energy
    mar_energy[i] = np.square(np.linalg.norm(s[:i])/np.linalg.norm(s[:i+1]))
    if np.square(np.linalg.norm(s[:i])/np.linalg.norm(s)) > 0.95:
        cutoff = i
        print "90 percent Cutoff at EV no. " + str(i+1)
        print "Percentage energy : " + str(100*np.square(np.linalg.norm(s[:i])/np.linalg.norm(s)))
        break

'''
fig, ax = plt.subplots(nrows=2, ncols=2)

plt.subplot(2,2,1)
plt.plot(vh[0,:], vh[1,:], 'yo')
plt.xlabel("Largest EV")
plt.ylabel("Second largest EV")
plt.title("Reduced bill space")
plt.subplot(2,2,2)
plt.plot(u[:,0], u[:,1], 'go')
plt.xlabel("Largest EV")
plt.ylabel("Second largest EV")
plt.title("Reduced member space")
plt.show()
plt.subplot(2,2,3)
plt.plot(s, 'r^')
plt.title("Spectrum of A")
plt.subplot(2,2,4)
plt.plot(rel_energy, 'go')
plt.title("Relative spectrum of A")
'''
'''
plt.subplot(2,2,4)
plt.plot(mar_energy[:cutoff], 'y^')
plt.title("Marginal spectrum of A")
plt.show()
'''

rep_indices = []
dem_indices = []
other_indices = []
party = ['Other' for i in range(0, member_count)]
name = ['Unknown' for i in range(0, member_count)]
with open(data_folder+congress+'_members.csv') as csv_file:
    i = -1
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        if not (i == -1):
            if int(row[6]) == 100:
                dem_indices.append(i)
                party[i] = 'democrat'
            elif int(row[6]) == 200:
                rep_indices.append(i)
                party[i] = 'republican'
            else:
                other_indices.append(i)
            name[i] = row[9]
        i = i + 1

print "Number of Republicans : " + str(len(rep_indices))
print "Number of Democrats : " + str(len(dem_indices))
print "Number of Others : " + str(len(other_indices))

result = ['Unkown' for i in range(0, bill_count)]
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
        i = i + 1

print "Number of Passed bills : " + str(len(passed_indices))
print "Number of Failed bills : " + str(len(failed_indices))
print "Number of Undecided bills : " + str(len(undecided_indices))

with open(output_folder+'eigenmembers.csv', 'w') as csvfile:
    data = zip(name, party, u[:,0], u[:,1])
    writercsv = csv.writer(csvfile)
    for row in data:
        writercsv.writerow(row)
with open(output_folder+'eigenbills.csv', 'w') as csvfile:
    data = zip(result, vh[0,:], vh[1,:])
    writercsv = csv.writer(csvfile)
    for row in data:
        writercsv.writerow(row)

custom_lines = [Line2D([0], [0], color='red', linestyle='None', marker='o'),
                Line2D([0], [0], color='blue', linestyle='None', marker='o'),
                Line2D([0], [0], color='green', linestyle='None', marker='o')]

fig, ax = plt.subplots()

ax.legend(custom_lines, ['Republican', 'Democrat', 'Undetermined'])

plt.plot([u[i,0] for i in other_indices], [u[i,1] for i in other_indices], 'go')
plt.plot([u[i,0] for i in rep_indices], [u[i,1] for i in rep_indices], 'ro')
plt.plot([u[i,0] for i in dem_indices], [u[i,1] for i in dem_indices], 'bo')
plt.title("Reduced member space")
plt.savefig(output_folder+"eigenmembers.png")
#plt.show()

custom_lines = [Line2D([0], [0], color='green', linestyle='None', marker='o'),
                Line2D([0], [0], color='red', linestyle='None', marker='o'),
                Line2D([0], [0], color='yellow', linestyle='None', marker='o')]

fig, ax = plt.subplots()

ax.legend(custom_lines, ['Passed', 'Failed', 'Undetermined'])

plt.plot([vh[0,i] for i in undecided_indices], [vh[1,i] for i in undecided_indices], 'yo')
plt.plot([vh[0,i] for i in passed_indices], [vh[1,i] for i in passed_indices], 'go')
plt.plot([vh[0,i] for i in failed_indices], [vh[1,i] for i in failed_indices], 'ro')
plt.title("Reduced bill space")
plt.savefig(output_folder+"eigenbills.png")
#plt.show()

fig = plt.figure()
ax = Axes3D(fig)
#ax.scatter([vh[0,i] for i in range(0, bill_count)], [vh[1,i] for i in range(0, bill_count)], [vh[2,i] for i in range(0, bill_count)])
ax.scatter([vh[0,i] for i in passed_indices], [vh[1,i] for i in passed_indices], [vh[2,i] for i in passed_indices], c='g')
ax.scatter([vh[0,i] for i in failed_indices], [vh[1,i] for i in failed_indices], [vh[2,i] for i in failed_indices], c='r')
ax.scatter([vh[0,i] for i in undecided_indices], [vh[1,i] for i in undecided_indices], [vh[2,i] for i in undecided_indices], c='y')
plt.title("3 Dom. EVs in policy space")
#plt.show()

n_bins = 40

# We can set the number of bins with the `bins` kwarg
fig, axs = plt.subplots(2, 2, figsize=(20,10))

plt.title("Distribution of EVs")
# We can set the number of bins with the `bins` kwarg
plt.subplot(2,2,1)
plt.hist(u[:,0], bins=n_bins)
#plt.title("Distribution of Largest EVs in member space")
plt.xlabel("Largest EV in member space")
plt.subplot(2,2,2)
plt.hist(u[:,1], bins=n_bins)
#plt.title("Distribution of second largest EVs in member space")
plt.xlabel("Second largest EV in member space")
plt.subplot(2,2,3)
plt.hist(vh[0,:], bins=n_bins)
#plt.title("Distribution of Largest EVs in bill space")
plt.xlabel("Largest EV in bill space")
plt.subplot(2,2,4)
plt.hist(vh[1,:], bins=n_bins)
#plt.title("Distribution of second largest EVs in bill space")
plt.xlabel("Second largest EV in bill space")
#plt.figure(figsize=(20,10))
plt.savefig(output_folder+"evdistribution.png")
#plt.show()
