import os
import numpy as np

stage = 2 # which layer
index = 14 # which pdb
head = 1 # which head

dataset = np.load('case_study/case_study_stage_' + str(stage) + '_dataset.npy', allow_pickle=True)
print(dataset.shape)
print(dataset[index])
pdb_name = dataset[index][0][0:4]
chain_name = dataset[index][0][-1]
print([pdb_name, chain_name])
max_len = min(1024, int(dataset[index][1]))
print(max_len)
min_att = min(dataset[index][2][head])
max_att = max(dataset[index][2][head])
print([min_att, max_att])

partner_chain_name = 'B'
mut_chain_name = 'A'
ssmall = 25
smax = 80

cmd = open('case_study/cmds/' + str(index) + '_' + pdb_name + '_' + chain_name + '_cmds.txt','w+')
cmd.write('remove resn HOH;\n')
cmd.write('select chain ' + chain_name + ';\n')
cmd.write('show surface, all;\n')
cmd.write('color gray90, sele;\n')
cmd.write('set transparency,0.15;\n')
cmd.write('select chain ' + mut_chain_name + ';\n')
cmd.write('color palegreen, sele;\n')

prefix = '/' + pdb_name + '/B/' + chain_name + '/'
offset = 1

for res in range(max_len):
    if(dataset[index][2][head][res] < ssmall):
        cmd.write('select ' + prefix + str(res + offset) + ";\n")
        cmd.write('color gray90, sele;\n')
    elif(dataset[index][2][head][res] >= smax):
        cmd.write('select ' + prefix + str(res + offset) + ";\n")
        cmd.write('color density, sele;\n')        
    else:
        cmd.write('select ' + prefix + str(res + offset) + ";\n")
        cmd.write('color lightblue, sele;\n')        