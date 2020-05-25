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
with open(data_folder+'eigenbills.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        eigenbills.append(map(float, row[1:3]))
        result.append(row[0])
        polarization.append(float(row[3]))
bill_count = len(eigenbills)

eps = 0.01
def proj_span(theta, phi):
    A = np.matrix([[math.cos(theta), math.cos(phi)],[math.sin(theta), math.sin(phi)]])
    if abs(np.linalg.det(A)) < eps:
        return 10000
    Ainv = np.matrix([[math.sin(phi), -math.cos(phi)],[-math.sin(theta), math.cos(theta)]])/np.linalg.det(A)
    #Ainv = np.linalg.inv(A)
    c = [eigenbills[i][0]*Ainv[0,0]+eigenbills[i][1]*Ainv[0,1] for i in range(0, bill_count)]
    return max(c)-min(c)

def proj_span_2(rho):
    if abs(math.tan(rho)) < eps:
        return 10000
    Ainv = np.matrix([[1, 0],[1,-1/math.tan(rho)]])
    #Ainv = np.linalg.inv(A)
    c = [x[i]*Ainv[1,0]+y[i]*Ainv[1,1] for i in range(0, bill_count)]
    return max(c)-min(c)


N=50
#graph = [[proj_span(i*2*math.pi/N,j*2*math.pi/N) for i in range(0,N)] for j in range(0,N])
graph = [[proj_span(i*math.pi/N,j*math.pi/N) for i in range(0,N)] for j in range(0,N)]
minval = min([min(graph[i][:]) for i in range(0,N)])
#minval = 0.06
print minval
argmin = []
for i in range(0,N):
    for j in range(0,N):
        #if abs(graph[i]-minval) < eps and graph[i+1] > graph[i] and graph[i-1] > graph[i]:
        #if graph[i+1] > graph[i] and graph[i-1] > graph[i]:
        if graph[i][j] == minval:
            argmin.append([i,j])
#argmin = [[36.0,86.0]]
print argmin
theta = argmin[0][0]*math.pi/N
phi = argmin[0][1]*math.pi/N
print theta
print phi
print (phi-theta)/math.pi
A = np.matrix([[math.cos(theta), math.cos(phi)],[math.sin(theta), math.sin(phi)]])
print np.linalg.det(A)
#A = np.matrix([[1,2],[3,4]])
Ainv = np.matrix([[math.sin(phi), -math.cos(phi)],[-math.sin(theta), math.cos(theta)]])/np.linalg.det(A)
x = [Ainv[0,0]*eigenbills[i][0] + Ainv[0,1]*eigenbills[i][1] for i in range(0, bill_count)]
y = [Ainv[1,0]*eigenbills[i][0] + Ainv[1,1]*eigenbills[i][1] for i in range(0, bill_count)]
print A
#plt.plot(x,y,'yo')
#plt.show()
graph = [proj_span_2(i*math.pi/N) for i in range(0,N)]
minval = min(graph)
print minval
argmin = []
for i in range(0,N):
    if graph[i] == minval:
        argmin.append(i)
print argmin
rho = argmin[0]*math.pi/N
print rho
#plt.plot(x,y,'yo')
#plt.plot([i*minval*math.cos(rho)/N for i in range(0,N)],[i*minval*math.sin(rho)/N for i in range(0,N)], 'ro')
#plt.show()
B = np.matrix([[1, -math.cos(rho)/math.sin(rho)],[0, 1/math.sin(rho)]])
print B
print B[0,1]
#B = np.matrix([[1, 0],[math.cos(rho), math.sin(rho)]])
print np.linalg.det(B)
x2 = [B[0,0]*x[i] + B[0,1]*y[i] for i in range(0, bill_count)]
y2 = [B[1,0]*x[i] + B[1,1]*y[i] for i in range(0, bill_count)]
with open(output_folder+'eigenbills_squared.csv', 'w') as csvfile:
    data = zip(x2, y2)
    writercsv = csv.writer(csvfile)
    for row in data:
        writercsv.writerow(row)
#plt.plot(x2,y2,'yo')
#plt.show()
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
plt.savefig(output_folder+"eigenbills_squared_pf.png")
#plt.show()
plt.close(fig)


'''
Eigenbills colored by polarization
'''
colormap = [(float(polarization[i])/max(polarization),0.1,0.1) for i in range(0, bill_count)]
colormap = [((-0.6*polarization[i])/max(polarization) + 0.9, (-0.6*polarization[i])/max(polarization) + 0.9, (-0.6*polarization[i])/max(polarization) + 0.9) for i in range(0, bill_count)]
#colormap = [(float(polarization[i])/max(polarization),float(polarization[i])/max(polarization),float(polarization[i])/max(polarization)) for i in range(0, bill_count)]
fig, ax = plt.subplots()
plt.scatter(x2[:], y2[:], c=colormap)
plt.title("Eigenvectors in bill space colored by polarization")
plt.savefig(output_folder+"eigenbills_squared_polarized.png")
#plt.show()
plt.close(fig)
