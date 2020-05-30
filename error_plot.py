import csv
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D

import math



dem_vote = []
rep_vote = []
dem_vote_app = []
rep_vote_app = []
error_dem = []
error_rep = []


for i in range(95, 99):
    congress = 'H0'+str(i)
    data_folder = 'output/raw/'+congress+'/'+congress+'_'
    output_folder = 'output/raw/'+congress+'/'+congress+'_'
    with open(data_folder+'eigenbills_squared_normalized.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            dem_vote.append(float(row[0]))
            rep_vote.append(float(row[1]))
            dem_vote_app.append(float(row[2]))
            rep_vote_app.append(float(row[3]))
    #error.append([i, 100.0*sum([(dem_vote[i]-dem_vote_app[i])*(dem_vote[i]-dem_vote_app[i])+(rep_vote[i]-rep_vote_app[i])*(rep_vote[i]-rep_vote_app[i]) for i in range(0, len(dem_vote))])/sum([dem_vote[i]*dem_vote[i]+rep_vote[i]*rep_vote[i] for i in range(0, len(dem_vote))])])
    error_dem.append([i, 100.0*sum([(dem_vote[j]-dem_vote_app[j])*(dem_vote[j]-dem_vote_app[j]) for j in range(0, len(dem_vote))])/sum([dem_vote[j]*dem_vote[j] for j in range(0, len(dem_vote))])])
    error_rep.append([i, 100.0*sum([(rep_vote[j]-rep_vote_app[j])*(rep_vote[j]-rep_vote_app[j]) for j in range(0, len(dem_vote))])/sum([rep_vote[j]*rep_vote[j] for j in range(0, len(dem_vote))])])
    dem_vote = []
    rep_vote = []
    dem_vote_app = []
    rep_vote_app = []

    
for i in range(100, 117):
    congress = 'H'+str(i)
    data_folder = 'output/raw/'+congress+'/'+congress+'_'
    output_folder = 'output/raw/'+congress+'/'+congress+'_'
    with open(data_folder+'eigenbills_squared_normalized.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            dem_vote.append(float(row[0]))
            rep_vote.append(float(row[1]))
            dem_vote_app.append(float(row[2]))
            rep_vote_app.append(float(row[3]))
    #error.append([i, 100.0*sum([(dem_vote[i]-dem_vote_app[i])*(dem_vote[i]-dem_vote_app[i])+(rep_vote[i]-rep_vote_app[i])*(rep_vote[i]-rep_vote_app[i]) for i in range(0, len(dem_vote))])/sum([dem_vote[i]*dem_vote[i]+rep_vote[i]*rep_vote[i] for i in range(0, len(dem_vote))])])
    error_dem.append([i, 100.0*sum([(dem_vote[j]-dem_vote_app[j])*(dem_vote[j]-dem_vote_app[j]) for j in range(0, len(dem_vote))])/sum([dem_vote[j]*dem_vote[j] for j in range(0, len(dem_vote))])])
    error_rep.append([i, 100.0*sum([(rep_vote[j]-rep_vote_app[j])*(rep_vote[j]-rep_vote_app[j]) for j in range(0, len(dem_vote))])/sum([rep_vote[j]*rep_vote[j] for j in range(0, len(dem_vote))])])
    dem_vote = []
    rep_vote = []
    dem_vote_app = []
    rep_vote_app = []

custom_lines = [Line2D([0], [0], color='m', linestyle='None', marker='o'),
                Line2D([0], [0], color='y', linestyle='None', marker='o')]
fig, ax = plt.subplots()
plt.plot([error_dem[i][0] for i in range(0, len(error_dem))], [error_dem[i][1] for i in range(0, len(error_dem))], 'm')
plt.plot([error_dem[i][0] for i in range(0, len(error_dem))], [error_dem[i][1] for i in range(0, len(error_dem))], 'bo')
plt.plot([error_rep[i][0] for i in range(0, len(error_rep))], [error_rep[i][1] for i in range(0, len(error_rep))], 'y')
plt.plot([error_rep[i][0] for i in range(0, len(error_rep))], [error_rep[i][1] for i in range(0, len(error_rep))], 'ro')
ax.legend(custom_lines, ['Democrat', 'Republican'])
plt.ylabel("Error in percentage")
plt.xlabel("Congress")
plt.title("Relative L2 error in sum of Dem and Rep votes and squared dominant eigenvectors")
plt.savefig('output/error.png')
plt.show()
