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
    congress = 'H100'

data_folder = 'output/raw/'+congress+'/'+congress+'_'
output_folder = 'output/raw/'+congress+'/'+congress+'_'
hd_file = 'output/house_details.csv'

eigenbills = []
result = []
polarization = []
polarization_signed = []
rep_vote = []
dem_vote = []
with open(data_folder+'eigenbills.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        eigenbills.append(map(float, row[1:3]))
        result.append(row[0])
        polarization_signed.append(float(row[3]))
        polarization.append(abs(float(row[3])))
        rep_vote.append(float(row[5]))
        dem_vote.append(float(row[6]))
bill_count = len(eigenbills)
eigenmembers = []
party = []
with open(data_folder+'eigenmembers.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        eigenmembers.append(map(float, row[2:4]))
        party.append(row[1])
member_count = len(eigenmembers)

ev_1 = 0
ev_2 = 0
num_reps = 0
num_dems = 0

with open(hd_file) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        if row[0] == congress:
            ev_1 = float(row[1])
            ev_2 = float(row[2])
            num_reps = float(row[3])
            num_dems = float(row[4])

eps = 0.01

def proj_span_x(theta, phi):
    A = np.matrix([[math.cos(theta), math.cos(phi)],[math.sin(theta), math.sin(phi)]])
    if abs(np.linalg.det(A)) < eps:
        return 10000
    Ainv_1 = np.matrix([[math.sin(phi), -math.cos(phi)],[-math.sin(theta), math.cos(theta)]])/np.linalg.det(A)
    c = [eigenbills[i][0]*Ainv_1[0,0]+eigenbills[i][1]*Ainv_1[0,1] for i in range(0, bill_count)]
    return max(c)-min(c)

def proj_span_y(rho):
    if abs(math.tan(rho)) < eps:
        return 10000
    Ainv_2 = np.matrix([[1, 0],[1,-1/math.tan(rho)]])
    c = [x[i]*Ainv_2[1,0]+y[i]*Ainv_2[1,1] for i in range(0, bill_count)]
    return max(c)-min(c)

N=50

graph = [[proj_span_x(i*math.pi/N,j*math.pi/N) for i in range(0,N)] for j in range(0,N)]
minval = min([min(graph[i][:]) for i in range(0,N)])
argmin = []
for i in range(0,N):
    for j in range(0,N):
        if graph[i][j] == minval:
            argmin.append([i,j])
theta = argmin[0][0]*math.pi/N
phi = argmin[0][1]*math.pi/N
A = np.matrix([[math.cos(theta), math.cos(phi)],[math.sin(theta), math.sin(phi)]])
Ainv = np.matrix([[math.sin(phi), -math.cos(phi)],[-math.sin(theta), math.cos(theta)]])/np.linalg.det(A)
x = [Ainv[0,0]*eigenbills[i][0] + Ainv[0,1]*eigenbills[i][1] for i in range(0, bill_count)]
y = [Ainv[1,0]*eigenbills[i][0] + Ainv[1,1]*eigenbills[i][1] for i in range(0, bill_count)]

graph = [proj_span_y(i*math.pi/N) for i in range(0,N)]
minval = min(graph)
argmin = []
for i in range(0,N):
    if graph[i] == minval:
        argmin.append(i)
rho = argmin[0]*math.pi/N
B = np.matrix([[1, -math.cos(rho)/math.sin(rho)],[0, 1/math.sin(rho)]])
x2 = [B[0,0]*x[i] + B[0,1]*y[i] for i in range(0, bill_count)]
y2 = [B[1,0]*x[i] + B[1,1]*y[i] for i in range(0, bill_count)]

scalex = max(max(x2),-min(x2))
scaley = max(max(y2),-min(y2))

x2 = [x2[i]/scalex for i in range(0, bill_count)]
y2 = [y2[i]/scaley for i in range(0, bill_count)]

scale = np.matrix([[1/scalex, 0],[0,1/scaley]])
lt_final = scale*B*Ainv
print "Squaring linear transformation :"
print scale*B*Ainv
with open(output_folder+'eigenbills_squared.csv', 'w') as csvfile:
    data = zip(x2, y2)
    writercsv = csv.writer(csvfile)
    for row in data:
        writercsv.writerow(row)

passed_indices = []
failed_indices = []
undecided_indices = []
for i in range(0, bill_count):
    if result[i].lower() == "passed":
        passed_indices.append(i)
    elif result[i].lower() == "failed":
        failed_indices.append(i)
    else:
        undecided_indices.append(i)

dem_indices = []
rep_indices = []
other_indices = []
for i in range(0, member_count):
    if party[i].lower() == "democrat":
        dem_indices.append(i)
    elif party[i].lower() == "republican":
        rep_indices.append(i)
    else:
        other_indices.append(i)

'''
Eigenbills colored by pass/fail plot
'''
custom_lines = [Line2D([0], [0], color='green', linestyle='None', marker='o'),
                Line2D([0], [0], color='red', linestyle='None', marker='o'),
                Line2D([0], [0], color='yellow', linestyle='None', marker='o')]
fig, ax = plt.subplots()
ax.legend(custom_lines, ['Passed', 'Failed', 'Undetermined'])
plt.plot([x2[i] for i in undecided_indices], [y2[i] for i in undecided_indices], 'yo')
plt.plot([x2[i] for i in passed_indices], [y2[i] for i in passed_indices], 'go')
plt.plot([x2[i] for i in failed_indices], [y2[i] for i in failed_indices], 'ro')
plt.title("Eigenvectors in bill space colored by result")
#plt.savefig(output_folder+"eigenbills_squared_pf.png")
#plt.show()
plt.close(fig)


'''
Eigenbills colored by polarization
'''
colormap = [(float(polarization[i])/max(polarization),0.1,0.1) for i in range(0, bill_count)]
colormap = [((-0.6*polarization[i])/max(polarization) + 0.9, (-0.6*polarization[i])/max(polarization) + 0.9, (-0.6*polarization[i])/max(polarization) + 0.9) for i in range(0, bill_count)]
fig, ax = plt.subplots()
plt.scatter(x2[:], y2[:], c=colormap)
plt.title("Eigenvectors in bill space colored by polarization")
#plt.savefig(output_folder+"eigenbills_squared_polarized.png")
#plt.show()
plt.close(fig)

mxr = 0
myr = 0
mxd = 0
myd = 0
sq_r = 0
sq_d = 0
sq_x = 0
sq_y = 0
for i in range(0, bill_count):
    mxd = mxd + x2[i]*dem_vote[i]
    mxr = mxr + x2[i]*rep_vote[i]
    myr = myr + y2[i]*rep_vote[i]
    myd = myd + y2[i]*dem_vote[i]
    sq_r = sq_r + rep_vote[i]*rep_vote[i]
    sq_d = sq_d + dem_vote[i]*dem_vote[i]
    sq_x = sq_x + x2[i]*x2[i]
    sq_y = sq_y + y2[i]*y2[i]

sq_r = math.sqrt(sq_r)
sq_d = math.sqrt(sq_d)
sq_x = math.sqrt(sq_x)
sq_y = math.sqrt(sq_y)

memspace_lt = lt_final*np.matrix([[1/ev_1,0],[0,1/ev_2]])
if abs(mxd/(sq_d*sq_x)) < abs(myd/(sq_d*sq_y)):
    multd = myd/(sq_d*sq_d)
    multr = mxr/(sq_r*sq_r)
    dem_ax = y2
    rep_ax = x2
    dem_ax_m = [memspace_lt[1,0]*eigenmembers[i][0]+memspace_lt[1,1]*eigenmembers[i][1] for i in range(0, member_count)]
    rep_ax_m = [memspace_lt[0,0]*eigenmembers[i][0]+memspace_lt[0,1]*eigenmembers[i][1] for i in range(0, member_count)]
    if multd < 0:
        multd = -multd
        for i in range(0, member_count):
           dem_ax_m[i] = -1*dem_ax_m[i]
        for i in range(0, bill_count):
           dem_ax[i] = -1*dem_ax[i]
    if multr < 0:
        multr = -multr
        for i in range(0, member_count):
           rep_ax_m[i] = -1*rep_ax_m[i]
        for i in range(0, bill_count):
           rep_ax[i] = -1*rep_ax[i]
else:
    multd = mxd/(sq_d*sq_d)
    multr = myr/(sq_r*sq_r)
    dem_ax = x2
    rep_ax = y2
    dem_ax_m = [memspace_lt[0,0]*eigenmembers[i][0]+memspace_lt[0,1]*eigenmembers[i][1] for i in range(0, member_count)]
    rep_ax_m = [memspace_lt[1,0]*eigenmembers[i][0]+memspace_lt[1,1]*eigenmembers[i][1] for i in range(0, member_count)]
    if multd < 0:
        multd = -multd
        for i in range(0, member_count):
           dem_ax_m[i] = -1*dem_ax_m[i]
        for i in range(0, bill_count):
           dem_ax[i] = -1*dem_ax[i]
    if multr < 0:
        multr = -multr
        for i in range(0, member_count):
           rep_ax_m[i] = -1*rep_ax_m[i]
        for i in range(0, bill_count):
           rep_ax[i] = -1*rep_ax[i]

app_num_reps = abs((max(rep_ax)-min(rep_ax))/(2*multr))
app_num_dems = abs((max(dem_ax)-min(dem_ax))/(2*multd))
print "Approx. republicans : " + str(app_num_reps)
print "Actual republicans : " + str(num_reps)
print "Approx. democrats : " + str(app_num_dems)
print "Actual democrats : " + str(num_dems)

fig, ax = plt.subplots()
plt.plot([multd*dem_vote[i] for i in range(0, bill_count)], [multr*rep_vote[i] for i in range(0, bill_count)], 'go')
plt.plot(dem_ax, rep_ax, 'yo')
plt.xlabel("Democratic axis")
plt.ylabel("Republican axis")
plt.savefig(output_folder+"eigenbills_squared_comparison.png")
#plt.show()
plt.close(fig)


plt.plot([dem_ax_m[i] for i in other_indices], [rep_ax_m[i] for i in other_indices], 'go')
plt.plot([dem_ax_m[i] for i in rep_indices], [rep_ax_m[i] for i in rep_indices], 'ro')
plt.plot([dem_ax_m[i] for i in dem_indices], [rep_ax_m[i] for i in dem_indices], 'bo')
plt.xlabel("Democratic axis")
plt.ylabel("Republican axis")
plt.savefig(output_folder+"eigenmembers_squared.png")
#plt.show()
plt.close(fig)
