import os
import numpy as np
'''
with open('H_sapiens_interfacesHQ_cleaning_all.txt', 'w+') as file_mapping_all:
    with open('H_sapiens_interfacesHQ.txt') as interface:
        line = interface.readline()
        while line:
            if(line.split()[3] != '[]' and line.split()[4] != '[]'):
                file_mapping_all.write(line)
            line = interface.readline()
      '''      
with open('H_sapiens_interfacesHQ_cleaning_pdb.txt', 'w+') as file_mapping_pdb:
    with open('H_sapiens_interfacesHQ.txt') as interface:
        line = interface.readline()
        while line:
            if(line.split()[3] != '[]' and line.split()[4] != '[]' and line.split()[2] == 'PDB'):
                file_mapping_pdb.write(line)
            line = interface.readline()