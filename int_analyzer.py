import csv
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D

import sys
import math

if len(sys.argv) >= 2:
    congress = sys.argv[1]
else:
    congress = 'H100'

data_folder = 'output/raw/'+congress+'/'+congress+'_'
output_folder = 'output/raw/'+congress+'/'+congress+'_'
write_folder = 'output/raw/'+congress+'/'+'interactive_output/'
hd_file = 'output/house_details.csv'

dem_ax_m = []
rep_ax_m = []
party = []
dem_indices = []
rep_indices = []
other_indices = []
member_details = []
with open(output_folder+'eigenmembers_squared_normalized.csv') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=',')
    for row in csv_reader:
        dem_ax_m.append(float(row[0]))
        rep_ax_m.append(float(row[1]))
with open(output_folder+'eigenmembers.csv') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=',')
    for row in csv_reader:
        party.append(row[1])
        member_details.append(row)
for i in range(0, len(party)):
    if party[i].lower() == 'republican':
        rep_indices.append(i)
    elif party[i].lower() == 'democrat':
        dem_indices.append(i)
    else:
        other_indices.append(i)

dem_ax = []
rep_ax = []
result = []
passed_indices = []
failed_indices = []
unknown_indices = []
bill_details = []
with open(output_folder+'eigenbills_squared_normalized.csv') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=',')
    for row in csv_reader:
        dem_ax.append(float(row[2]))
        rep_ax.append(float(row[3]))
with open(output_folder+'eigenbills.csv') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=',')
    for row in csv_reader:
        result.append(row[0])
        bill_details.append(row)
for i in range(0, len(result)):
    if result[i].lower() == 'failed':
        failed_indices.append(i)
    elif result[i].lower() == 'passed':
        passed_indices.append(i)
    else:
        unknown_indices.append(i)

def update_annot(ind, stat='p'):
    if stat == 'p':
        pos = sc_pass.get_offsets()[ind["ind"][0]]
        annot.xy = pos
        if (pos[0], pos[1]) in zip([dem_ax[i] for i in passed_indices], [rep_ax[i] for i in passed_indices]):
            zipped = list(zip([dem_ax[i] for i in passed_indices], [rep_ax[i] for i in passed_indices]))
            index = zipped.index((pos[0], pos[1]))
            print(bill_details[passed_indices[index]])
            text = "Description: " + bill_details[passed_indices[index]][-2] + ", " + bill_details[passed_indices[index]][-1] + "\nStatus: " + str((bill_details[passed_indices[index]][0])) + ", Votes: " + str((bill_details[passed_indices[index]][4:-2]))
            text = "Description: " + bill_details[passed_indices[index]][-1] + "\nStatus: " + str((bill_details[passed_indices[index]][0])) + ", Votes: " + str((bill_details[passed_indices[index]][4:-1]))
    if stat == 'f':
        print(ind)
        pos = sc_fail.get_offsets()[ind["ind"][0]]
        annot.xy = pos
        if (pos[0], pos[1]) in zip([dem_ax[i] for i in failed_indices], [rep_ax[i] for i in failed_indices]):
            zipped = list(zip([dem_ax[i] for i in failed_indices], [rep_ax[i] for i in failed_indices]))
            index = zipped.index((pos[0], pos[1]))
            print(bill_details[failed_indices[index]])
            text = "Description: " + bill_details[failed_indices[index]][-2] + ", " + bill_details[failed_indices[index]][-1] + "\nStatus: " + str((bill_details[failed_indices[index]][0])) + ", Votes: " + str((bill_details[failed_indices[index]][4:-2]))
            text = "Description: " + bill_details[failed_indices[index]][-1] + "\nStatus: " + str((bill_details[failed_indices[index]][0])) + ", Votes: " + str((bill_details[failed_indices[index]][4:-1]))
    if stat == 'u':
        pos = sc_unk.get_offsets()[ind["ind"][0]]
        annot.xy = pos
        if (pos[0], pos[1]) in zip([dem_ax[i] for i in unknown_indices], [rep_ax[i] for i in unknown_indices]):
            zipped = list(zip([dem_ax[i] for i in unknown_indices], [rep_ax[i] for i in unknown_indices]))
            index = zipped.index((pos[0], pos[1]))
            print(bill_details[unknown_indices[index]])
            text = "Description: " + bill_details[unknown_indices[index]][-2] + ", " + bill_details[unknown_indices[index]][-1] + "\nStatus: " + str((bill_details[unknown_indices[index]][0])) + ", Votes: " + str((bill_details[unknown_indices[index]][4:-2]))
            text = "Description: " + bill_details[unknown_indices[index]][-1] + "\nStatus: " + str((bill_details[unknown_indices[index]][0])) + ", Votes: " + str((bill_details[unknown_indices[index]][4:-1]))
    #sc_pass = plt.scatter([dem_ax[i] for i in passed_indices], [rep_ax[i] for i in passed_indices], c='g')
    annot.set_text(text)
    if stat == 'p':
        annot.get_bbox_patch().set_facecolor('g')
    elif stat == 'f':
        annot.get_bbox_patch().set_facecolor('r')
    elif stat == 'u':
        annot.get_bbox_patch().set_facecolor('y')
    #annot.get_bbox_patch().set_alpha(0.4)

def hover(event):
    vis = annot.get_visible()
    if event.inaxes == ax:
        cont_p, ind_p = sc_pass.contains(event)
        cont_f, ind_f = sc_fail.contains(event)
        cont_u, ind_u = sc_unk.contains(event)
        if cont_p:
            update_annot(ind_p, 'p')
            annot.set_visible(True)
            fig.canvas.draw_idle()
        elif cont_f:
            update_annot(ind_f, 'f')
            annot.set_visible(True)
            fig.canvas.draw_idle()
        elif cont_u:
            update_annot(ind_u, 'u')
            annot.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if vis:
                annot.set_visible(False)
                fig.canvas.draw_idle()

xmin = 0
xmax = 0
ymin = 0
ymax = 0
ans = 'n'
selected_indices = []
while 1 > 0:
    selected_indices = []
    plt.clf()
    mode = input("Do you want to analyze members or bills [m/b]:")
    if len(mode) == 0:
        print("Enter valid input [m/b]")
        break

    if mode[0] == 'm':
        plt.plot([dem_ax_m[i] for i in dem_indices], [rep_ax_m[i] for i in dem_indices], 'bo')
        plt.plot([dem_ax_m[i] for i in rep_indices], [rep_ax_m[i] for i in rep_indices], 'ro')
        plt.plot([dem_ax_m[i] for i in other_indices], [rep_ax_m[i] for i in other_indices], 'go')
        plt.xlabel("Democratic axis")
        plt.ylabel("Republican axis")
        plt.draw()
        plt.pause(1)
        pts = []
        plt.title("Select 2 corners of the rectangle")
        #fig, ax = plt.subplots()
        pts = np.asarray(plt.ginput(2, timeout=-1))
        xmin = min(pts[:,0])
        xmax = max(pts[:,0])
        ymin = min(pts[:,1])
        ymax = max(pts[:,1])
        pts = np.asarray([[xmin, ymin],[xmax,ymin],[xmax,ymax],[xmin,ymax]])
        ph = plt.fill(pts[:,0], pts[:,1], 'y', lw=2, alpha=0.5)
        plt.draw()
        plt.pause(0.001)
        while 1 > 0:
            ans = input("Happy with the rectangle? [y/n]")
            if ans[0].lower() == 'y':
                file_name = input("Enter the filename you wish to save the details in:")
                for i in range(0, len(dem_ax_m)):
                    if dem_ax_m[i] <= xmax and dem_ax_m[i] >= xmin and rep_ax_m[i] <= ymax and rep_ax_m[i] >= ymin:
                        selected_indices.append(i)
                with open(write_folder+file_name+'.csv','w') as csvfile:
                    writercsv = csv.writer(csvfile)
                    for i in selected_indices:
                        myCsvRow = member_details[i]
                        writercsv.writerow(myCsvRow)
                break
            elif ans[0].lower() == 'n':
                print("Ok, Starting over")
                break
            else:
                print("Enter valid input [y/n]")

    elif mode[0] == 'b':
        fig, ax = plt.subplots()
        annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
        annot.set_visible(False)
        sc_pass = plt.scatter([dem_ax[i] for i in passed_indices], [rep_ax[i] for i in passed_indices], c='g', s=10, alpha=0.5)
        sc_fail = plt.scatter([dem_ax[i] for i in failed_indices], [rep_ax[i] for i in failed_indices], c='r', s=10, alpha=0.5)
        sc_unk = plt.scatter([dem_ax[i] for i in unknown_indices], [rep_ax[i] for i in unknown_indices], c='y', s=10, alpha=0.5)
        plt.xlabel("Democratic axis")
        plt.ylabel("Republican axis")
        fig.canvas.mpl_connect("motion_notify_event", hover)
        plt.show()
        plt.pause(1)
        pts = []
        plt.title("Select 2 corners of the rectangle")
        #fig, ax = plt.subplots()
        pts = np.asarray(plt.ginput(2, timeout=-1))
        xmin = min(pts[:,0])
        xmax = max(pts[:,0])
        ymin = min(pts[:,1])
        ymax = max(pts[:,1])
        pts = np.asarray([[xmin, ymin],[xmax,ymin],[xmax,ymax],[xmin,ymax]])
        ph = plt.fill(pts[:,0], pts[:,1], 'y', lw=2, alpha=0.5)
        plt.draw()
        plt.pause(0.001)
        while 1 > 0:
            ans = input("Happy with the rectangle? [y/n]")
            if ans[0].lower() == 'y':
                file_name = input("Enter the filename you wish to save the details in:")
                for i in range(0, len(dem_ax_m)):
                    if dem_ax_m[i] <= xmax and dem_ax_m[i] >= xmin and rep_ax_m[i] <= ymax and rep_ax_m[i] >= ymin:
                        selected_indices.append(i)
                with open(write_folder+file_name+'.csv','w') as csvfile:
                    writercsv = csv.writer(csvfile)
                    for i in selected_indices:
                        myCsvRow = bill_details[i]
                        writercsv.writerow(myCsvRow)
                break
            elif ans[0].lower() == 'n':
                print("Ok, Starting over")
                break
            else:
                print("Enter valid input [y/n]")


    else:
        print("Entered unkown reply, quitting.")
        break
