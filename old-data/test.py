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

import csv
from collections import OrderedDict

row_count = 0
col_count = 0
with open('rollcall_h2019.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        row_count = row_count + 1
        col_count = len(row)

leg_index_csv = [i for i in range(0, row_count)]
leg_lnames_csv = ['a' for i in range(0, row_count)]
leg_state_csv = ['None' for i in range(0, row_count)]
row_count = 0
row_temp = []
with open('rollcall_h2019.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        row_temp = row[1:]
        leg_lnames_csv[row_count] = row[0].split(' ',1)[0]
        if len(row[0].split(' ',1)) > 1:
            leg_state_csv[row_count] = row[0].split(' ',1)[1]
        row_count = row_count + 1
legs_csv = list(zip(leg_index_csv,leg_lnames_csv, leg_state_csv))
print legs_csv
info = []
thisdict = dict()
thisdict['last_name'] = legs_csv[0][1]
thisdict['rollcall'] = row_temp
thisdict['state'] = legs_csv[0][2]

#thisdict = {'last_name': legs_csv[0][1], 'rollcall': row_temp, 'state': legs_csv[0][2]}
print thisdict
info.append(thisdict)
print info
with open('users.yaml', 'w') as f:
    data = yaml.dump(info, f, default_flow_style=False, sort_keys=False)

