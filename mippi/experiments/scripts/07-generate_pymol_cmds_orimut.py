import os
import numpy as np

stage = 2 # which layer
index = 14 # which pdb
head = 1 # which head

dataset = np.load('case_study/case_study_stage_' + str(stage) + '_mut_dataset.npy', allow_pickle=True)
print(dataset.shape)
print(dataset[index])
pdb_name = dataset[index][0][0:4]
chain_name = dataset[index][0][-1]
print([pdb_name, chain_name])
mut_pos = dataset[index][2]
print(mut_pos) #26
min_att = min(dataset[index][3][head])
max_att = max(dataset[index][3][head])
print([min_att, max_att])

ssmall = 3
sbig = 4
smax = 5

cmd = open('case_study/cmds/' + str(index) + '_' + pdb_name + '_' + chain_name + '_mut_cmds.txt','w+')

cmd.write('select chain ' + chain_name + ';\n')
cmd.write('show surface, sele;\n')
cmd.write('color palegreen, sele;\n')
cmd.write('set transparency, 0.15;\n')

prefix = '/' + pdb_name + '/A/' + chain_name + '/'
cmd.write('select ' + prefix + str(mut_pos) + ";\n")
cmd.write('color limon, sele;\n')
'''
offset = 1
for res in range(mut_pos - 25, mut_pos + 25):
    if(dataset[index][3][head][res] < ssmall):
        cmd.write('select ' + prefix + str(res + offset) + ";\n")
        cmd.write('color limon, sele;\n')
    elif(dataset[index][3][head][res] >= ssmall and dataset[index][3][head][res] <= sbig):
        cmd.write('select ' + prefix + str(res + offset) + ";\n")
        cmd.write('color lime, sele;\n')
    elif(dataset[index][3][head][res] >= smax):
        cmd.write('select ' + prefix + str(res + offset) + ";\n")
        cmd.write('color forest, sele;\n')        
    else:
        cmd.write('select ' + prefix + str(res + offset) + ";\n")
        cmd.write('color green, sele;\n') 
            '''
