import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D

import sys
import math



k = 5
spec_mat = [[-1 for j in range(95, 118)] for i in range(k)]
A_orig = [np.zeros((1,1)) for j in range(95, 118)]
A_proj_1 = [np.zeros((1,1)) for j in range(95, 118)]
A_proj_2 = [np.zeros((1,1)) for j in range(95, 118)]
A_diff_1 = [np.zeros((1,1)) for j in range(95, 118)]
A_diff_2 = [np.zeros((1,1)) for j in range(95, 118)]
for num in range(95, 118):
    if num < 100:
        congress = 'H0' + str(num)
    else:
        congress = 'H'+str(num)

    data_folder = 'data/'+congress+'/'
    output_folder = 'output/eigenmat/'
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
    A_orig[num-95] = np.copy(A)
    A_sorted = np.zeros(A.shape)
    u, s, vh = np.linalg.svd(A)
    '''
    a1 = 0;
    a2 = 1;
    b1 = 0;
    b2 = 1;
    '''
    a1 = 1;
    a2 = 0;
    b1 = 1;
    b2 = 0;
    mem_inds = sorted(range(u.shape[0]), key=lambda k: b1*u[k,0]+b2*u[k,1])
    bill_inds = sorted(range(vh.shape[0]), key=lambda k: a1*vh[0,k]+a2*vh[1,k])
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            A_sorted[i,j] = A_orig[num-95][mem_inds[i], bill_inds[j]]
    A_orig[num-95] = np.copy(A_sorted)
    A = np.copy(A_sorted)
    u, s, vh = np.linalg.svd(A)
    
    print(A.shape)
    print(u.shape)
    print(vh.shape)
    A_proj_1[num-95] = np.zeros(A.shape)
    A_proj_2[num-95] = np.zeros(A.shape)
    #for i in range(s.shape[0]):
    A_proj_1[num-95] = s[0]*np.outer(u[:,0], vh[0,:])
    for i in range(2):
        A_proj_2[num-95] = A_proj_2[num-95] + s[i]*np.outer(u[:,i], vh[i,:])
    A_diff_1[num-95] = A_orig[num-95]-A_proj_1[num-95]
    A_diff_2[num-95] = A_orig[num-95]-A_proj_2[num-95]
    
    '''
    plt.imshow(A_orig[num-95])
    plt.xlabel('Bills')
    plt.ylabel('Members')
    plt.title('Original Bill matrix H'+str(num))
    plt.savefig(output_folder+'separated/H'+str(num)+'_orig')
    plt.imshow(A_proj_1[num-95])
    plt.xlabel('Bills')
    plt.ylabel('Members')
    plt.title('Projected(1EV) Bill matrix H'+str(num))
    plt.savefig(output_folder+'separated/H'+str(num)+'_proj_1')
    plt.imshow(A_diff_1[num-95])
    plt.xlabel('Bills')w
    plt.ylabel('Members')
    plt.title('Difference(1EV) Bill matrix H'+str(num))
    plt.savefig(output_folder+'separated/H'+str(num)+'_diff_1')
    plt.imshow(A_proj_2[num-95])
    plt.xlabel('Bills')
    plt.ylabel('Members')
    plt.title('Projected(2EVs) Bill matrix H'+str(num))
    plt.savefig(output_folder+'separated/H'+str(num)+'_proj_2')
    plt.imshow(A_diff_2[num-95])
    plt.xlabel('Bills')
    plt.ylabel('Members')
    plt.title('Difference(2EVs) Bill matrix H'+str(num))
    plt.savefig(output_folder+'separated/H'+str(num)+'_diff_2')

    fig, axs = plt.subplots(3)
    fig.suptitle('Original/Projected/Difference Bill matrix H' +str(num))
    axs[0].imshow(A_orig[num-95])
    axs[1].imshow(A_proj_1[num-95])
    axs[2].imshow(A_diff_1[num-95])
    plt.xlabel('Bills')
    plt.ylabel('Members')
    plt.savefig(output_folder+'H'+str(num)+'_projected')
    plt.clf()
    '''
    
    '''
    fig, axs = plt.subplots(3)
    fig.suptitle('Leading Member/Bill Eigenvector and Projected Bill matrix H' +str(num))
    #axs[0].imshow(np.tile(u[:,0].reshape((len(mem_inds), 1)), (len(mem_inds), 100)))
    #axs[0].imshow(np.tile(u[mem_inds,0], (len(mem_inds), 10)))
    #axs[1].imshow(vh[0, bill_inds].reshape((len(bill_inds), 1)))
    #axs[1].imshow(np.tile(vh[0,:].reshape((len(bill_inds), 1)), (len(bill_inds), 100)))
    #axs[1].imshow(np.tile(vh[0, bill_inds], (len(bill_inds), 10)))
    axs[0].plot([np.sqrt(s[0])*u[i, 0] for i in range(len(mem_inds))])
    axs[0].set_ylim(-1.5,1.5)
    axs[1].plot([np.sqrt(s[0])*vh[0, i] for i in range(len(bill_inds))])
    axs[1].set_ylim(-1.5,1.5)
    axs[2].imshow(A_proj_1[num-95])
    plt.xlabel('Bills')
    plt.ylabel('Members')
    plt.savefig(output_folder+'H'+str(num)+'_split')
    plt.clf()
    '''
    
    '''
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot([np.sqrt(s[0])*u[i, 0] for i in range(len(mem_inds))])
    axs[0, 0].set_ylim(-1.5,1.5)
    axs[1, 0].plot([np.sqrt(s[0])*vh[0, i] for i in range(len(bill_inds))])
    axs[1, 0].set_ylim(-1.5,1.5)
    axs[0, 1].hist([np.sqrt(s[0])*u[i, 0] for i in range(len(mem_inds))], bins=[-1.5 + 0.05*i for i in range(61)], orientation='horizontal')
    axs[1, 1].hist([np.sqrt(s[0])*vh[i, 0] for i in range(len(bill_inds))], bins=[-1.5 + 0.05*i for i in range(61)], orientation='horizontal')
    '''
    fig = plt.figure()
    fig.suptitle('Leading Member/Bill Eigenvector and Projected Bill matrix H' +str(num))
    spec = mpl.gridspec.GridSpec(ncols=8, nrows=3)
    ax1 = fig.add_subplot(spec[0,0:4])
    ax2 = fig.add_subplot(spec[0,5:])
    ax3 = fig.add_subplot(spec[1,0:4])
    ax4 = fig.add_subplot(spec[1,5:])
    ax5 = fig.add_subplot(spec[2,2:6])
    ax1.plot([np.sqrt(s[0])*u[i, 0] for i in range(len(mem_inds))])
    ax1.set_ylim(-1.5,1.5)
    ax3.plot([np.sqrt(s[0])*vh[0, i] for i in range(len(bill_inds))])
    ax3.set_ylim(-1.5,1.5)
    width = 0.1
    ax2.hist([np.sqrt(s[0])*u[i, 0] for i in range(len(mem_inds))], bins=[-1.5 + width*i for i in range(int(3.0/width)+1)], orientation='horizontal')
    ax4.hist([np.sqrt(s[0])*vh[0, i] for i in range(len(bill_inds))], bins=[-1.5 + width*i for i in range(int(3.0/width)+1)], orientation='horizontal')
    ax5.imshow(A_proj_1[num-95])
    #plt.savefig(output_folder+'H'+str(num)+'_hist')
    plt.clf()

    fig = plt.figure()
    #fig.suptitle('Leading Member/Bill Eigenvector and Projected Bill matrix H' +str(num))
    spec = mpl.gridspec.GridSpec(ncols=1, nrows=2)
    ax2 = fig.add_subplot(spec[0,0])
    ax4 = fig.add_subplot(spec[1,0])
    width = 0.1
    ax2.hist([np.sqrt(s[0])*u[i, 0] for i in range(len(mem_inds))], bins=[-1.5 + width*i for i in range(int(3.0/width)+1)], orientation='horizontal')
    ax4.hist([np.sqrt(s[0])*vh[0, i] for i in range(len(bill_inds))], bins=[-1.5 + width*i for i in range(int(3.0/width)+1)], orientation='horizontal')
    #plt.savefig(output_folder+'H'+str(num)+'_histonl')
    plt.clf()
    

    width = 0.1
    
    list_plot = [np.sqrt(s[0])*u[i, 0] for i in range(len(mem_inds))]
    l_max = max(list_plot)
    l_min = min(list_plot)
    list_plot = [2.0*(x-l_min)/(l_max-l_min) - 1.0 for x in list_plot]

    plt.hist(list_plot, bins=[-1.5 + width*i for i in range(int(3.0/width)+1)], orientation='horizontal')
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.savefig(output_folder+'H'+str(num)+'_hist_mems_norm')
    plt.clf()

    list_plot = [np.sqrt(s[0])*vh[0, i] for i in range(len(bill_inds))]
    l_max = max(list_plot)
    l_min = min(list_plot)
    list_plot = [2.0*(x-l_min)/(l_max-l_min) - 1.0 for x in list_plot]
    plt.hist(list_plot, bins=[-1.5 + width*i for i in range(int(3.0/width)+1)], orientation='horizontal')
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.savefig(output_folder+'H'+str(num)+'_hist_bill_norm')
