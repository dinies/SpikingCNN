from os.path import dirname, realpath
import csv

sub_path = dirname(dirname(realpath(__file__)))
PATH= sub_path + '/models/datasets/'+'dummyDataset.csv'
FEATURE_NAMES= [
        'f1','f2','f3','f4','labels'
        ]

def writeFirstLine( filewriter, feature_names = FEATURE_NAMES):
    filewriter.writerow(feature_names)

def writeALine( filewriter, line):
    filewriter.writerow(line)

with open(PATH, 'a') as csvfile:
    filewriter=csv.writer(csvfile,delimiter=',',quotechar='|',quoting=csv.QUOTE_MINIMAL)
    writeFirstLine( filewriter)

    writeALine( filewriter, [ '0','0','0','0','1'])
    writeALine( filewriter, [ '0','0','0','0','1'])
    writeALine( filewriter, [ '0','0','0','0','1'])
    writeALine( filewriter, [ '0','0','0','0','1'])
    writeALine( filewriter, [ '0','0','0','0','1'])
    writeALine( filewriter, [ '0','0','0','0','1'])
    writeALine( filewriter, [ '0','0','0','0','1'])
