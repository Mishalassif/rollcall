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

eps = 0.01
def proj_span(theta, phi):
    A = np.matrix([[math.cos(theta), math.cos(phi)],[math.sin(theta), math.sin(phi)]])
    if abs(np.linalg.det(A)) < eps:
        return 10000
    Ainv_1 = np.matrix([[math.sin(phi), -math.cos(phi)],[-math.sin(theta), math.cos(theta)]])/np.linalg.det(A)
    c = [eigenbills[i][0]*Ainv_1[0,0]+eigenbills[i][1]*Ainv_1[0,1] for i in range(0, bill_count)]
    return max(c)-min(c)

def proj_span_2(rho):
    if abs(math.tan(rho)) < eps:
        return 10000
    Ainv_2 = np.matrix([[1, 0],[1,-1/math.tan(rho)]])
    c = [x[i]*Ainv_2[1,0]+y[i]*Ainv_2[1,1] for i in range(0, bill_count)]
    return max(c)-min(c)

N=50
graph = [[proj_span(i*math.pi/N,j*math.pi/N) for i in range(0,N)] for j in range(0,N)]
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
graph = [proj_span_2(i*math.pi/N) for i in range(0,N)]
minval = min(graph)
argmin = []
for i in range(0,N):
    if graph[i] == minval:
        argmin.append(i)
rho = argmin[0]*math.pi/N
B = np.matrix([[1, -math.cos(rho)/math.sin(rho)],[0, 1/math.sin(rho)]])
x2 = [B[0,0]*x[i] + B[0,1]*y[i] for i in range(0, bill_count)]
y2 = [B[1,0]*x[i] + B[1,1]*y[i] for i in range(0, bill_count)]
print "Squaring linear transformation :"
print B*Ainv
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

mx = 0
my = 0
sq_r = 0
sq_d = 0
sq_x = 0
sq_y = 0
for i in range(0, bill_count):
    #mx = mx + x2[i]*dem_vote[i]
    mx = mx + x2[i]*rep_vote[i]
    #my = my + y2[i]*rep_vote[i]
    my = my + y2[i]*dem_vote[i]
    sq_r = sq_r + rep_vote[i]*rep_vote[i]
    sq_d = sq_d + dem_vote[i]*dem_vote[i]
    sq_x = sq_x + x2[i]*x2[i]
    sq_y = sq_y + y2[i]*y2[i]

sq_r = math.sqrt(sq_r)
sq_d = math.sqrt(sq_d)
sq_x = math.sqrt(sq_x)
sq_y = math.sqrt(sq_y)

#print mx/(sq_d*sq_x)
print mx/(sq_r*sq_x)
#print my/(sq_r*sq_y)
print my/(sq_d*sq_y)

#multx = mx/(sq_d*sq_d)
multx = mx/(sq_r*sq_r)
#multy = my/(sq_r*sq_r)
multy = my/(sq_d*sq_d)
print "Approx. republicans : " + str((max(x2)-min(x2))/(2*multx))
print "Approx. democrats : " + str((max(y2)-min(y2))/(2*multy))

#plt.plot([multx*dem_vote[i] for i in range(0, bill_count)], [multy*rep_vote[i] for i in range(0, bill_count)], 'go')
#plt.plot([multx*rep_vote[i] for i in range(0, bill_count)], [multy*dem_vote[i] for i in range(0, bill_count)], 'go')
#plt.plot(x2, y2, 'yo')
#plt.savefig(output_folder+"eigenbills_squared_comparison.png")
#plt.show()
