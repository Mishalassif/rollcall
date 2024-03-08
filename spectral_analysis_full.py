import csv
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D

import sys
import math

import scipy.stats

if len(sys.argv) == 2:
    congress = sys.argv[1]
else:
    congress = 'H115'

data_folder = 'data/'+congress+'/'
output_folder = 'output/eigenval/'+congress+'_ev'
output_cum_folder = 'output/eigenval/cum/'+'full_ev'
unnorm_output_folder = 'output/eigenval/unnormalized/'+congress+'_ev'
res1_output_folder = 'output/eigenval/res1/'+congress+'_ev'
res2_output_folder = 'output/eigenval/res2/'+congress+'_ev'
save_specs = True

mp_output_folder = 'output/eigenval/mp/'+congress+'_mp'
save_mp = False

hd_file = 'output/house_details.csv'
trunc = 2

congresses = []
congresses = congresses+['H0'+str(x) for x in range(95,100)]
congresses = congresses+['H'+str(x) for x in range(100,118)]

con_c = 0
for congress in congresses:
    con_c=con_c+1
    data_folder = 'data/'+congress+'/'
    output_folder = 'output/eigenval/'+congress+'_ev'
    output_cum_folder = 'output/eigenval/cum/'+congress+'_ev'
    unnorm_output_folder = 'output/eigenval/unnormalized/'+congress+'_ev'
    res1_output_folder = 'output/eigenval/res1/'+congress+'_ev'
    res2_output_folder = 'output/eigenval/res2/'+congress+'_ev'
    save_specs = True

    mp_output_folder = 'output/eigenval/mp/'+congress+'_mp'
    save_mp = False

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

    u, s, vh = np.linalg.svd(A)
    u, s_c, vh = np.linalg.svd(A/np.linalg.norm(A))
    s_c_cum = [x*x for x in s_c]
#for i in range(len(s_c_cum)-1):
#    s_c_cum[-1-i-1] += s_c_cum[-1-i] 
    for i in range(1, len(s_c_cum)):
        s_c_cum[i] += s_c_cum[i-1] 


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

    if save_specs == True:
        colors = ((con_c/25.0), 0, 1-(con_c/25.0))
        print(colors)            
        plt.plot(s_c_cum, c=colors, marker='^', markersize=0.5) # alpha=min(0.1+(float(i)/24.0),1))#, markersize=0.5)
#plt.plot(s_c/np.linalg.norm(s_c), 'r^')
        #plt.title("Normalized Spectrum of Bill-Member Matrix ("+congress+")")
        plt.xlim(-10, 500)
        plt.ylim(0, 1.05)
        #plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
plt.savefig(output_cum_folder+'_full_normalized')
plt.show()
