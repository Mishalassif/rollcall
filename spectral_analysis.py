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
output_cum_folder = 'output/eigenval/cum/'+congress+'_ev'
unnorm_output_folder = 'output/eigenval/unnormalized/'+congress+'_ev'
res1_output_folder = 'output/eigenval/res1/'+congress+'_ev'
res2_output_folder = 'output/eigenval/res2/'+congress+'_ev'
save_specs = True

mp_output_folder = 'output/eigenval/mp/'+congress+'_mp'
save_mp = False

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
    plt.plot(s, 'r^')
    plt.title("Spectrum of Bill-Member Matrix ("+congress+")")
    plt.xlim(-10, 500)
    plt.ylim(0, 1500)
    #plt.savefig(unnorm_output_folder)
    plt.clf()

    plt.plot(s_c, 'r^')
#plt.plot(s_c/np.linalg.norm(s_c), 'r^')
    plt.title("Normalized Spectrum of Bill-Member Matrix ("+congress+")")
    plt.xlim(-10, 500)
    plt.ylim(0, 1)
    #plt.savefig(output_folder+'_normalized')
    plt.clf()

    plt.plot(s_c_cum, 'r^')
#plt.plot(s_c/np.linalg.norm(s_c), 'r^')
    #plt.title("Normalized Spectrum of Bill-Member Matrix ("+congress+")")
    plt.xlim(-10, 500)
    plt.ylim(0, 1.05)
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.savefig(output_cum_folder+'_normalized')
    plt.clf()

    plt.hist(s_c[2:], bins=50, density=1)
#plt.plot(s_c/np.linalg.norm(s_c), 'r^')
    plt.title("Histogram of residual eigenvalues ("+congress+")")
#plt.xlim(-10, 500)
#plt.ylim(0, 1)
    #plt.savefig(res2_output_folder)
    plt.clf()

    plt.hist(s_c[1:], bins=50, density=1)
#plt.plot(s_c/np.linalg.norm(s_c), 'r^')
    plt.title("Histogram of residual eigenvalues ("+congress+")")
#plt.xlim(-10, 500)
#plt.ylim(0, 1)
    #plt.savefig(res1_output_folder)
    plt.clf()

    plt.plot(np.sort(s_c[2:]), np.linspace(0, 1, len(s_c[2:]), endpoint=False))
    plt.title("CDF of 2+ residual eigenvalues")
    plt.xlim(0, 0.2)
    plt.ylim(0, 1)
    #plt.savefig(res2_output_folder + "_cdf")
    plt.clf()

    plt.plot(np.sort(s_c[1:]), np.linspace(0, 1, len(s_c[1:]), endpoint=False))
    plt.title("CDF of 1+ residual eigenvalues")
    plt.xlim(0, 0.6)
    plt.ylim(0, 1)
    #plt.savefig(res1_output_folder + "_cdf")
    plt.clf()


'''
Marchenko-Pastur Tests
'''

from skrmt.ensemble.spectral_law import MarchenkoPasturDistribution

x1 = np.linspace(0, 4, num=1000)
x2 = np.linspace(0, 5, num=2000)


#for ratio in [0.2, 0.4, 0.6, 1.0, 1.4]:
'''
for ratio in [0.6, 0.8, 1.0]:
    mpl = MarchenkoPasturDistribution(beta=1, ratio=ratio, sigma=1.0)

    y1 = mpl.pdf(x1)
    y2 = mpl.cdf(x2)

    ax1.plot(x1, y1, label=f"$\lambda$ = {ratio} ")
    ax2.plot(x2, y2, label=f"$\lambda$ = {ratio} ")
'''
h = 0.025
N = 40
if save_mp == True:
    for k in range(1,4):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))
        spec_norm = len(s_c[k:])*s_c[k:]/np.sum(s_c[k:])
        ratio_o = round(np.std(spec_norm)*np.std(spec_norm), 2)

        ks_stat = []
        p_val = []
        for ratio in [h*i for i in range(1,N+1)]:
            mpl = MarchenkoPasturDistribution(beta=1, ratio=ratio, sigma=1.0)
            res = scipy.stats.kstest(spec_norm, mpl.cdf)
            #print('lambda: ' + str(ratio))
            ks_stat.append(res.statistic)
            p_val.append(res.pvalue)
            if int(ratio*N)%(int(N/4)) == 0:
                print(str(int(ratio*N)) +'\% done')

        ks_stat = np.array(ks_stat)
        p_val = np.array(p_val)
        min_pos = ks_stat.argmin()
        ratio_ks = round((min_pos+1)*h, 3)
        print('Best p value: ' + str(p_val[min_pos]))

        for ratio in [ratio_o, ratio_ks]:
            mpl = MarchenkoPasturDistribution(beta=1, ratio=ratio, sigma=1.0)

            y1 = mpl.pdf(x1)
            y2 = mpl.cdf(x2)
            
            if ratio == ratio_o:
                #ax1.plot(x1, y1, label=f"$\lambda$ = {ratio} (matching moment)")
                #ax2.plot(x2, y2, label=f"$\lambda$ = {ratio} (matching moment)")
                print('')
            else:
                ax1.plot(x1, y1, label=f"$\lambda$ = {ratio} (min ks-statistic)")
                ax2.plot(x2, y2, label=f"$\lambda$ = {ratio} (min ks-statistic)")

        ax1.hist(spec_norm, bins='auto', density=1)
        ax1.legend()
        ax1.set_xlim(-0.5, 6)
        ax1.set_xlabel("x", fontweight="bold")
        ax1.set_ylabel("density", fontweight="bold")

        ax2.plot(np.sort(spec_norm), np.linspace(0, 1, len(spec_norm), endpoint=False))
        ax2.legend()
        ax2.set_ylim(0, 1.1)
        ax2.set_xlim(0, 5)
        ax2.set_xlabel("x", fontweight="bold")
        ax2.set_ylabel("CDF", fontweight="bold")

        fig.suptitle("Marchenko-Pastur distribution vs Empirical singular value distribution (>"+str(k)+" EVs)", fontweight="bold")
        #plt.savefig(mp_output_folder+'_'+str(k))
        plt.clf()
