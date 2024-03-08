import csv
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D

import sys
import math

interactive_mode = False
if len(sys.argv) >= 2:
    congress = sys.argv[1]
    if len(sys.argv) == 3:
        interactive_mode = False
else:
    congress = 'H100'

data_folder = 'output/raw/'+congress+'/'+congress+'_'
#output_folder = 'output/raw/'+congress+'/'+congress+'_'
output_folder = 'output/eigenbills_density/'+congress+'_'
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
        #writercsv.writerow(myCsvRow)
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
print("Squaring linear transformation :")
print(scale*B*Ainv)

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
plt.plot([x2[i] for i in undecided_indices], [y2[i] for i in undecided_indices], 'yo', alpha=0.1)
plt.plot([x2[i] for i in passed_indices], [y2[i] for i in passed_indices], 'go', alpha=0.1)
plt.plot([x2[i] for i in failed_indices], [y2[i] for i in failed_indices], 'ro', alpha=0.1)
plt.title("Eigenvectors in bill space colored by result")
if interactive_mode == False:
    plt.savefig(output_folder+"eigenbills_squared_pf.png")
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

#temp = lt_final*np.matrix([[1/ev_1,0],[0,1/ev_2]])
memspace_lt = np.transpose(np.linalg.inv(lt_final))*np.matrix([[ev_1,0],[0,ev_2]])
#memspace_lt = np.linalg.inv(temp)
print(memspace_lt)
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
print("Approx. republicans : " + str(app_num_reps))
print("Actual republicans : " + str(num_reps))
print("Approx. democrats : " + str(app_num_dems))
print("Actual democrats : " + str(num_dems))

plt.plot([dem_ax_m[i] for i in other_indices], [rep_ax_m[i] for i in other_indices], 'go', alpha=0.1)
plt.plot([dem_ax_m[i] for i in rep_indices], [rep_ax_m[i] for i in rep_indices], 'ro', alpha=0.1)
plt.plot([dem_ax_m[i] for i in dem_indices], [rep_ax_m[i] for i in dem_indices], 'bo', alpha=0.1)
#plt.xlabel("Democratic axis")
#plt.ylabel("Republican axis")
if interactive_mode == False:
    plt.savefig(output_folder+"eigenmembers_squared.png")
#plt.show()
plt.close(fig)


data_folder = 'data/'+congress+'/'

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

print(A)
print("Number of votes : " + str(vote_count))
print("Number of abstentions : " + str(abstention_count))

yn_vote_count = 0
for i in range(0, member_count):
    for j in range(0, bill_count):
        yn_vote_count = yn_vote_count + abs(A[i][j])

print("Fraction of votes : " + str(float(vote_count)/(member_count*bill_count)))
print((yn_vote_count)/vote_count)
print((yn_vote_count)/(member_count*bill_count))
print((yn_vote_count+abstention_count)/vote_count)
print((yn_vote_count+abstention_count)/(member_count*bill_count))

trunc = 2
u, s, vh = np.linalg.svd(A)
smat = np.zeros((member_count, bill_count))
smat[:member_count, :member_count] = np.diag(s)
A_d = (u[:,:trunc].dot(smat[:trunc,:trunc].dot(vh[:trunc,:])))
rel_energy = np.copy(s)
mar_energy = np.copy(s)
total_energy = (np.linalg.norm(A))*(np.linalg.norm(A))

print("Max EVs : " + str(s[0]) + ", " + str(s[1]))
print("Approximation error(p) in l2 : " + str(100*np.linalg.norm(A_d-A)/np.linalg.norm(A)))

cutoff = 0
for i in range(0, len(s)-1):
    rel_energy[i] = 100*(s[i]*s[i])/total_energy
    mar_energy[i] = np.square(np.linalg.norm(s[:i])/np.linalg.norm(s[:i+1]))
    if np.square(np.linalg.norm(s[:i])/np.linalg.norm(s)) > 0.95:
        cutoff = i
        print("90 percent Cutoff at EV no. " + str(i+1))
        print("Percentage energy : " + str(100*np.square(np.linalg.norm(s[:i])/np.linalg.norm(s))))
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

print("Number of Republicans : " + str(len(rep_indices)))
print("Number of Democrats : " + str(len(dem_indices)))
print("Number of Others : " + str(len(other_indices)))


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

custom_lines = [Line2D([0], [0], color='green', linestyle='None', marker='o'),
                Line2D([0], [0], color='red', linestyle='None', marker='o'),
                Line2D([0], [0], color='yellow', linestyle='None', marker='o')]
fig, ax = plt.subplots()
#ax.legend(custom_lines, ['Passed', 'Failed', 'Undetermined'])
plt.plot([vh[0,i] for i in undecided_indices], [vh[1,i] for i in undecided_indices], 'yo', alpha=0.1)
plt.plot([vh[0,i] for i in passed_indices], [vh[1,i] for i in passed_indices], 'go', alpha=0.1)
plt.plot([vh[0,i] for i in failed_indices], [vh[1,i] for i in failed_indices], 'ro', alpha=0.1)
#plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
#plt.title("Eigenvectors in bill space colored by result")
plt.savefig(output_folder+"eigenbills_pf.png")
#plt.show()
plt.close(fig)

u, s, vh = np.linalg.svd(A)
#plt.plot([u[i,0] for i in other_indices], [u[i,1] for i in other_indices], 'go', alpha=0.1)
#plt.plot([u[i,0] for i in rep_indices], [u[i,1] for i in rep_indices], 'ro', alpha=0.1)
#plt.plot([u[i,0] for i in dem_indices], [u[i,1] for i in dem_indices], 'bo', alpha=0.1)
plt.plot([dem_ax_m[i] for i in other_indices], [rep_ax_m[i] for i in other_indices], 'go', alpha=0.1)
plt.plot([dem_ax_m[i] for i in rep_indices], [rep_ax_m[i] for i in rep_indices], 'ro', alpha=0.1)
plt.plot([dem_ax_m[i] for i in dem_indices], [rep_ax_m[i] for i in dem_indices], 'bo', alpha=0.1)
#plt.xlabel("Democratic axis")
#plt.ylabel("Republican axis")
#plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
if interactive_mode == False:
    plt.savefig(output_folder+"eigenmembers.png")
#plt.show()
plt.close(fig)

