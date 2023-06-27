import os

stage = 2 # which layer
index = 14 # which pdb
head = 2 # which head

partner_chain_name = 'B'
mut_chain_name = 'A'
length = 185

cmd = open('case_study/cmds/14_1E96_B_interface_cmds.txt','w+')
cmd.write('remove resn HOH;\n')
cmd.write('show surface, all;\n')
cmd.write('color gray90, all;\n')
cmd.write('set transparency,0.15;\n')
cmd.write('select chain A;\n')
cmd.write('hide everything, sele;\n')
prefix = '/1E96/B/B/'

interface_region = [(2.6,3.4),(35.6,36.4), (65.6,69.4),(101.6,102.4), (103.6,104.4),(105.6,108.4),(110.6,112.4)]

for i in range(length):
    flag = 0
    for region in interface_region:
        if(i > region[0] and i < region[1]):
            flag = 1
    if(flag == 1):
        cmd.write('select ' + prefix + str(i) + ";\n")
        cmd.write('color limon, sele;\n')        