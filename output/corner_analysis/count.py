import csv
import sys
if len(sys.argv) > 1:
    lower = int(sys.argv[1])
    upper = int(sys.argv[2])
    print(lower)
    print(upper)
else:
    lower = 115
    upper = 115

congresses = ['H'+str(i)+'_corner_bills.csv' for i in range(lower, upper+1)]

bills = dict()
for cngr in congresses:
    with open(cngr) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            bill = row[-1]
            bill = bill.replace(" ","").lower()
            if bill in bills:
                bills[bill] = bills[bill]+1
            else:
                bills[bill] = 1

print('\n')
print('========== MOST COMMON BILLS ===========')
print('\n')
bills = dict(sorted(bills.items(), key=lambda item: item[1]))
#print(bills)
#print(list(bills.items())[-10:])
#for bill in (list(bills.items())[-10:]):
#    print('   ' + str(bill))
bills = list(bills.items())
bills.reverse()
for bill in bills:
    if bill[1]<3:
        break
    print('   ' + str(bill))

print('\n')
print('========== MOST COMMON KEYWORDS ===========')
print('\n')
keywords = dict()
for cngr in congresses:
    with open(cngr) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            bill = row[-1]
            bill = bill.lower()
            keyword = bill.split()
            for word in keyword:
                word = word.replace(",","")
                if word in keywords:
                    keywords[word] = keywords[word]+1
                else:
                    keywords[word] = 1
keywords = dict(sorted(keywords.items(), key=lambda item: item[1]))
#print(list(keywords.items())[-100:])
count = 0
line_to_print = ''
rows=5
keywords = list(keywords.items())
keywords.reverse()
for word in keywords:
    if word[1] < 10:
        break
    if count >= 400:
        break
    line_to_print = line_to_print + '   ' + str(word)
    count = count+1
    if count%rows == 0:
        print(line_to_print)
        line_to_print = ''
