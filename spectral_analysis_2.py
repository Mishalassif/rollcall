import csv
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D

import sys
import math

from skrmt.ensemble.spectral_law import MarchenkoPasturDistribution
import scipy.stats

k = 5
spec_mat = [[-1 for j in range(95, 116)] for i in range(k)]

k_p = 3
p_vals = [[0.0 for j in range(95, 116)] for i in range(k_p)]
ks_stats = [[0.0 for j in range(95, 116)] for i in range(k_p)]
rat_act = [[0.0 for j in range(95, 116)] for i in range(k_p)]
for num in range(95, 116):
    if num < 100:
        congress = 'H0' + str(num)
    else:
        congress = 'H'+str(num)

    data_folder = 'data/'+congress+'/'
    output_folder = 'output/eigenval/'+'leading_evs'
    mp_folder = 'output/eigenval/'+'mp/p_values'

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
    print(float(member_count)/float(bill_count))
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

    for i in range(k):
        #spec_mat[i][num-95] = s[i]*s[i]/(np.linalg.norm(s)*np.linalg.norm(s))
        spec_mat[i][num-95] = s[i]*s[i]/(sum([s[i]*s[i] for i in range(len(s))]))

    u, s_c, vh = np.linalg.svd(A/np.linalg.norm(A))
    for i in range(k_p):
        spec_norm = len(s_c[k:])*s_c[k:]/np.sum(s_c[k:])
        ks_stat = []
        p_val = []
        for ratio in [0.05*i for i in range(1,21)]:
            mpl = MarchenkoPasturDistribution(beta=1, ratio=ratio, sigma=1.0)
            res = scipy.stats.kstest(spec_norm, mpl.cdf)
            #print('lambda: ' + str(ratio))
            ks_stat.append(res.statistic)
            p_val.append(res.pvalue)
        ks_stat = np.array(ks_stat)
        p_val = np.array(p_val)
        min_pos = ks_stat.argmin()
        ratio_ks = round((min_pos+1)*0.05,2)
        p_vals[i][num-95] = p_val[min_pos]
        ks_stats[i][num-95] = ratio_ks
        rat_act[i][num-95] = float(member_count)/float(bill_count)

fig, ax = plt.subplots()
for i in range(k):
    ax.plot([j for j in range(95,116)], spec_mat[i], label=str(i+1)+'th EV')
ax.plot([j for j in range(95,116)], [spec_mat[0][k]+spec_mat[1][k] for k in range(len(spec_mat[0]))], label='1+2 EVs')
ax.legend()
plt.title("Fraction of Leading Eigenvalues of spectrum of Bill-Member Matrix")
plt.xlim(90, 120)
plt.ylim(0, 1)
#plt.savefig(output_folder)
plt.clf()
#plt.show()
fig, ax = plt.subplots()
k_p = 1
for i in range(k_p):
    ax.plot([j for j in range(95,116)], rat_act[i], label='Actual m/n ratio')
ax.legend()
plt.title("Minimum ks-test p-value and ratio for residuals tested against Marchenko-Pastur")
plt.xlim(90, 120)
plt.ylim(0, 1)
#plt.savefig(mp_folder)
#plt.clf()
for i in range(k_p):
    ax.plot([j for j in range(95,116)], ks_stats[i], label='MP ratio for residual >' + str(i+1)+ 'EV')
    ax.plot([j for j in range(95,116)], p_vals[i], label='ks-test p-val for residual >' + str(i+1)+ 'EV')
ax.legend()
plt.savefig(mp_folder)
plt.clf()
