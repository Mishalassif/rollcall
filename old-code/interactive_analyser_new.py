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
        file_name = input("Enter the filename you wish to save the details in:")
        plt.plot([dem_ax[i] for i in passed_indices], [rep_ax[i] for i in passed_indices], 'go')
        plt.plot([dem_ax[i] for i in failed_indices], [rep_ax[i] for i in failed_indices], 'ro')
        plt.plot([dem_ax[i] for i in unknown_indices], [rep_ax[i] for i in unknown_indices], 'yo')
        plt.xlabel("Democratic axis")
        plt.ylabel("Republican axis")
        plt.draw()
        plt.pause(1)
        pts = []
        plt.title("Select 2 corners of the Upper left rectangle")
        plt.draw()
        pts = np.asarray(plt.ginput(2, timeout=-1))
        xmin = min(pts[:,0])
        xmax = max(pts[:,0])
        ymin = min(pts[:,1])
        ymax = max(pts[:,1])
        pts = np.asarray([[xmin, ymin],[xmax,ymin],[xmax,ymax],[xmin,ymax]])
        ph = plt.fill(pts[:,0], pts[:,1], 'y', lw=2, alpha=0.5)
        plt.title("Select 2 corners of the Lower left rectangle")
        plt.draw()
        plt.pause(0.001)
        while 1 > 0:
            #ans = input("Happy with the rectangle? [y/n]")
            ans = ['y']
            if ans[0].lower() == 'y':
                for i in range(0, len(dem_ax)):
                    if dem_ax[i] <= xmax and dem_ax[i] >= xmin and rep_ax[i] <= ymax and rep_ax[i] >= ymin:
                        selected_indices.append(i)
                with open(write_folder+file_name+'.csv','w') as csvfile:
                    writercsv = csv.writer(csvfile)
                    myCsvRow = ['Upper', 'left']
                    writercsv.writerow(myCsvRow)
                    for i in selected_indices:
                        myCsvRow = bill_details[i]
                        writercsv.writerow(myCsvRow)
                break
            elif ans[0].lower() == 'n':
                print("Ok, Starting over")
                break
            else:
                print("Enter valid input [y/n]")
        pts = np.asarray(plt.ginput(2, timeout=-1))
        xmin = min(pts[:,0])
        xmax = max(pts[:,0])
        ymin = min(pts[:,1])
        ymax = max(pts[:,1])
        pts = np.asarray([[xmin, ymin],[xmax,ymin],[xmax,ymax],[xmin,ymax]])
        ph = plt.fill(pts[:,0], pts[:,1], 'y', lw=2, alpha=0.5)
        plt.title("Select 2 corners of the Lower right rectangle")
        plt.draw()
        plt.pause(0.001)
        while 1 > 0:
            #ans = input("Happy with the rectangle? [y/n]")
            ans = ['y']
            if ans[0].lower() == 'y':
                selected_indices = []
                for i in range(0, len(dem_ax_m)):
                    if dem_ax_m[i] <= xmax and dem_ax_m[i] >= xmin and rep_ax_m[i] <= ymax and rep_ax_m[i] >= ymin:
                        selected_indices.append(i)
                with open(write_folder+file_name+'.csv','a') as csvfile:
                    writercsv = csv.writer(csvfile)
                    myCsvRow = ['Lower', 'left']
                    writercsv.writerow(myCsvRow)
                    for i in selected_indices:
                        myCsvRow = bill_details[i]
                        writercsv.writerow(myCsvRow)
                break
            elif ans[0].lower() == 'n':
                print("Ok, Starting over")
                break
            else:
                print("Enter valid input [y/n]")
        
        pts = np.asarray(plt.ginput(2, timeout=-1))
        xmin = min(pts[:,0])
        xmax = max(pts[:,0])
        ymin = min(pts[:,1])
        ymax = max(pts[:,1])
        pts = np.asarray([[xmin, ymin],[xmax,ymin],[xmax,ymax],[xmin,ymax]])
        ph = plt.fill(pts[:,0], pts[:,1], 'y', lw=2, alpha=0.5)
        plt.title("Select 2 corners of the Upper right rectangle")
        plt.draw()
        plt.pause(0.001)
        while 1 > 0:
            #ans = input("Happy with the rectangle? [y/n]")
            ans = ['y']
            if ans[0].lower() == 'y':
                selected_indices = []
                for i in range(0, len(dem_ax_m)):
                    if dem_ax_m[i] <= xmax and dem_ax_m[i] >= xmin and rep_ax_m[i] <= ymax and rep_ax_m[i] >= ymin:
                        selected_indices.append(i)
                with open(write_folder+file_name+'.csv','a') as csvfile:
                    writercsv = csv.writer(csvfile)
                    myCsvRow = ['Lower', 'right']
                    writercsv.writerow(myCsvRow)
                    for i in selected_indices:
                        myCsvRow = bill_details[i]
                        writercsv.writerow(myCsvRow)
                break
            elif ans[0].lower() == 'n':
                print("Ok, Starting over")
                break
            else:
                print("Enter valid input [y/n]")
        
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
            #ans = input("Happy with the rectangle? [y/n]")
            ans = ['y']
            if ans[0].lower() == 'y':
                selected_indices = []
                for i in range(0, len(dem_ax_m)):
                    if dem_ax_m[i] <= xmax and dem_ax_m[i] >= xmin and rep_ax_m[i] <= ymax and rep_ax_m[i] >= ymin:
                        selected_indices.append(i)
                with open(write_folder+file_name+'.csv','a') as csvfile:
                    writercsv = csv.writer(csvfile)
                    myCsvRow = ['Upper', 'right']
                    writercsv.writerow(myCsvRow)
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
