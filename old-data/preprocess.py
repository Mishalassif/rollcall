import argparse as ap, contextlib, fnmatch, os, sys, time, warnings, yaml

# version dependent libraries
# https://docs.python.org/2/library/urllib.html
# https://docs.python.org/3.0/library/urllib.parse.html
if (sys.version_info > (3, 0)):
    from urllib.request import urlopen
    import urllib.parse as urlparse
else:
    from urllib2 import urlopen
    import urlparse


import yaml
import csv

row_count = 0
col_count = 0
with open('rollcall_h2019.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        row_count = row_count + 1
        col_count = len(row)

leg_index_csv = [i for i in range(0, row_count)]
leg_lnames_csv = ['a' for i in range(0, row_count)]
leg_state_csv = ['Unknown' for i in range(0, row_count)]
row_count = 0
with open('rollcall_h2019.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        leg_lnames_csv[row_count] = row[0].split(' ',1)[0]
        if len(row[0].split(' ',1)) > 1:
            leg_state_csv[row_count] = row[0].split(' ',1)[1]
        row_count = row_count + 1
legs_csv = list(zip(leg_index_csv,leg_lnames_csv, leg_state_csv))

repeated_legs = []
parties = ['Unknown' for i in range(0,row_count)]
states = ['Unknown' for i in range(0,row_count)]
count = 0
matched_legs = 0
with open('legislators-current.yaml') as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    legislators = yaml.full_load(file)
    for leg_csv in legs_csv:
        repeat = 0
        count = 0
        for leg in legislators:
            if leg_csv[1] == leg['name']['last']:
                parties[leg_csv[0]] = leg['terms'][-1]['party']
                states[leg_csv[0]] = leg['terms'][-1]['state']
                if repeat == 0:
                    matched_legs= matched_legs+1
                repeat = repeat + 1
        if repeat > 1:
            parties[leg_csv[0]] = 'Unknown'
            states[leg_csv[0]] = 'Unknown'
            repeated_legs.append(leg_csv)
    repeated_parties = ['None' for i in range(0, len(repeated_legs))]
    for leg_csv in repeated_legs:
        for leg in legislators:
            if leg_csv[1] == leg['name']['last']:
                if leg_csv[2] == '('+leg['terms'][-1]['state']+')':
                    parties[leg_csv[0]] = leg['terms'][-1]['party']
                    states[leg_csv[0]] = leg['terms'][-1]['state']
rows = [[0 for i in range(0, col_count+1)] for j in range(0, row_count)]
with open('rollcall_h2019.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    i = 0
    for row in csv_reader:
        rows[i][0] = row[0]
        rows[i][1] = parties[i]
        rows[i][2:] = row[1:]
        i = i+1
with open('rollcall_h2019_parties.csv', 'w') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerows(rows)
