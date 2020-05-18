import csv
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D


row_count = 0
col_count = 0

trunc = (min(row_count, col_count))/2
trunc = 2
with open('rollcall_h2019_parties.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        row_int = map(int, row[2:])
        col_count = len(row_int)
        row_count = row_count + 1
    print row_count
    print col_count

party = ['Unknown' for i in range(0, row_count)]
name = ['Unknown' for i in range(0, row_count)]
A = np.zeros((row_count, col_count))
I = np.eye(col_count)

with open('rollcall_h2019_parties.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    i = 0
    for row in csv_reader:
        name[i] = row[0]
        party[i] = row[1]
        A[i][0:] = map(int, row[2:])
        i = i+1

u, s, vh = np.linalg.svd(A)
smat = np.zeros((row_count, col_count))
smat[:row_count, :row_count] = np.diag(s)
A_d = (u[:,:trunc].dot(smat[:trunc,:trunc].dot(vh[:trunc,:])))
rel_energy = np.copy(s)

print "Max EVs : " + str(s[0]) + ", " + str(s[1])
print "Approximation error(p) in l2 : " + str(100*np.linalg.norm(A_d-A)/np.linalg.norm(A))

cutoff = 0
for i in range(0, len(s)-1):
    rel_energy[i] = np.square(np.linalg.norm(s[:i])/np.linalg.norm(s[:i+1]))
    if np.square(np.linalg.norm(s[:i])/np.linalg.norm(s)) > 0.95:
        cutoff = i
        print "90 percent Cutoff at EV no. " + str(i+1)
        print "Percentage energy : " + str(100*np.square(np.linalg.norm(s[:i])/np.linalg.norm(s)))
        break


#print np.linalg.norm(A-A_d)
#print np.linalg.norm(A)

rep_indices = []
dem_indices = []
other_indices = []
for i in range(0, row_count):
    if party[i] == 'Republican':
        rep_indices.append(i)
    elif party[i] == 'Democrat':
        dem_indices.append(i)
    else:
        other_indices.append(i)

print "Number of Republicans : " + str(len(rep_indices))
print "Number of Democrats : " + str(len(dem_indices))
print "Number of Others : " + str(len(other_indices))

num_politicians = row_count


with open('eigenpoliticians.csv', 'w') as csvfile:
    data = zip(name, party, u[:,0], u[:,1])
    writercsv = csv.writer(csvfile)
    for row in data:
        writercsv.writerow(row)
with open('eigenpolicies.csv', 'w') as csvfile:
    data = zip(vh[0,:], vh[1,:])
    writercsv = csv.writer(csvfile)
    for row in data:
        writercsv.writerow(row)
'''
fig, ax = plt.subplots(nrows=2, ncols=1)

plt.subplot(2,2,1)
plt.plot(vh[0,:], vh[1,:], 'yo')
plt.xlabel("Largest EV")
plt.xlabel("Second largest EV")
plt.title("Dom. EVs in policy space")
plt.subplot(2,2,2)
plt.plot(u[:,0], u[:,1], 'go')
plt.xlabel("Largest EV")
plt.xlabel("Second largest EV")
plt.subplot(2,2,3)
plt.plot(s, 'r^')
plt.title("Spectrum of A")
plt.subplot(2,2,4)
plt.plot(rel_energy[:cutoff], 'y^')
plt.title("Relative spectrum of A")
plt.show()
'''
plt.plot(vh[0,:], vh[1,:], 'yo')
plt.xlabel("Largest EV")
plt.xlabel("Second largest EV")
plt.title("Dom. EVs in policy space")
plt.show()

plt.plot(vh[0,:], [0.5 for i in vh[0,:]], 'yo')
plt.xlabel("Largest EV")
plt.title("Largest EVs in policy space")
plt.show()

custom_lines = [Line2D([0], [0], color='red', linestyle='None', marker='o'),
                Line2D([0], [0], color='blue', linestyle='None', marker='o'),
                Line2D([0], [0], color='green', linestyle='None', marker='o')]

fig, ax = plt.subplots()

ax.legend(custom_lines, ['Republican', 'Democrat', 'Undetermined'])

plt.plot([u[i,0] for i in rep_indices], [u[i,1] for i in rep_indices], 'ro')
plt.plot([u[i,0] for i in dem_indices], [u[i,1] for i in dem_indices], 'bo')
plt.plot([u[i,0] for i in other_indices], [u[i,1] for i in other_indices], 'go')
plt.title("2 Dom. EVs in politician space")
#plt.plot([u[i,0] for i in other_indices], [u[i,1] for i in other_indices], 'go')
plt.show()
fig = plt.figure()
ax = Axes3D(fig)

ax.legend(custom_lines, ['Republican', 'Democrat', 'Undetermined'])

ax.scatter([u[i,0] for i in rep_indices], [u[i,1] for i in rep_indices], [u[i,2] for i in rep_indices], c='r')
ax.scatter([u[i,0] for i in dem_indices], [u[i,1] for i in dem_indices], [u[i,2] for i in dem_indices], c = 'b')
ax.scatter([u[i,0] for i in other_indices], [u[i,1] for i in other_indices], [u[i,2] for i in other_indices], c = 'g')
plt.title("3 Dom. EVs in politician space")
#plt.plot([u[i,0] for i in other_indices], [u[i,1] for i in other_indices], 'go')
plt.show()


fig = plt.figure()
ax = Axes3D(fig)
ax.scatter([vh[0,i] for i in range(0, num_politicians)], [vh[1,i] for i in range(0, num_politicians)], [vh[2,i] for i in range(0, num_politicians)])
plt.title("3 Dom. EVs in policy space")
plt.show()
