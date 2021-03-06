import os
import re
import logging
import sys
import subprocess
import multiprocessing
import numpy as np
import time
from tensorflow.keras.preprocessing.sequence import pad_sequences
import requests

class utils():
    def __init__(self, psiblast_bin=None, db_path=None, tmp_path='./mippi_tmp'):
        self.psiblast_bin = psiblast_bin
        self.db_path = db_path
        self.tmp_path = tmp_path
        if not os.path.isdir(self.tmp_path):
            os.makedirs(self.tmp_path)
        # self.input_path = input_path
        # self.output_path = output_path
        # self.aa = ['A','R','D','C','Q','E','H','I','G','N',
        #            'L','K','M','F','P','S','T','W','Y','V']  
    
    
    def splitItems(self, input_path):
        p1 = []
        p2 = []
        var = []
        with open(input_path, 'r') as f:
            counter = 0
            for line in f:
                counter += 1
                line = line.strip()
                line = line.split('\t')
                if len(line) == 3:
                    p1.append(line[0])
                    p2.append(line[1])
                    var.append(line[2])
                else:
                    logging.error('Expected 3 items in a line, but %d items found in line %d' % (len(line), counter))
                    sys.exit(1)
        return p1, p2, var


    def getFasta(self, id):
        fasta_inline = ''
        fasta = requests.get('https://www.uniprot.org/uniprot/%s.fasta' % str(id))
        if fasta.encoding == 'ISO-8859-1':
            input_flag = False
            for i in fasta.text:
                if i == '\n':
                    input_flag = True
                if input_flag and (i != '\n'):
                    fasta_inline += i
        else:
            fasta_inline = 'nan'
        
        return fasta_inline
    
    def transId2FastaFile(self, p1, p2, var):
        trans_file = os.path.join(self.tmp_path, 'transformed_file.txt')
        with open(trans_file, 'w') as f:
            for i in range(len(p1)):
                f.write(p1[i] + '\t' + p2[i] + '\t' + var[i] + '\n')
        print('ID file transformed into sequence file in %s' % trans_file)

        return trans_file

    def checkFormat(self, input_path):
        aalist = ['A','R','D','C','Q','E','H','I','G','N',
                  'L','K','M','F','P','S','T','W','Y','V'] 
        if not os.path.isfile(input_path):
            logging.error(str(input_path) + ' is not a available file')
            sys.exit(1)
            # return False
        else:
            p1_list = []
            p2_list = []
            ori_list = []
            mut_list = []
            pos_list = []
            # check valid lines
            check_list = []
            anno_list = []
            with open(input_path, 'r') as f:
                counter = 0
                for line in f:
                    # default set valid
                    check_line = True
                    anno_line = '.'

                    counter += 1
                    line = line.strip()
                    line = line.split('\t')
                    if len(line) != 3:
                        logging.error('Expected 3 items in a line, but %d items found in line %d' % (len(line), counter))
                        sys.exit(1)
                    
                    for j in range(2):
                        if line[j] == 'nan':
                            logging.error('Can not retrieve FASTA in line %d part %d from Uniprot.org' % (counter, j + 1))
                            check_line = False
                            anno_line = 'Can not retrieve FASTA in this line part %d from Uniprot.org' % (j + 1)

                    for k in range(2):
                        for i in range(len(line[k])):
                            if (not (line[k][i] in aalist)) and check_line:
                                # check AA letters in FASTA
                                logging.error('Unexpected letter %s found in line %d part %d position %d, which is not normal 20 amino acids.' % (line[k][i], counter, k + 1, i))
                                # sys.exit(1)
                                check_line = False
                                anno_line = 'Unexpected letter %s found in line, part %d position %d, which is not normal 20 amino acids.' % (line[k][i], k + 1, i)

                    if re.search(r'^[A-Z]+\d+[A-Z]+$', line[2]) and check_line:
                        ori, mut = re.findall(r'[A-Z]+', line[2])
                        pos = int(re.findall(r'\d+', line[2])[0])
                        # check AA letter in annotation original part
                        for i in ori:
                            if (not (i in aalist)) and check_line:
                                logging.error('Unexpected letter %s found in line %d annotation part, which is not normal 20 amino acids.' % (i, counter))
                                # sys.exit(1)
                                check_line = False
                                anno_line = 'Unexpected letter %s found in line %d annotation part, which is not normal 20 amino acids.' % (i, counter)
                        # check AA letter in annotation alternative part
                        for i in mut:
                            if (not (i in aalist)) and check_line:
                                logging.error('Unexpected letter %s found in line %d annotation part, which is not normal 20 amino acids.' % (i, counter))
                                # sys.exit(1)
                                check_line = False
                                anno_line = 'Unexpected letter %s found in line %d annotation part, which is not normal 20 amino acids.' % (i, counter)
                        # check annotation position within reference length
                        if (pos > len(line[0]) + len(ori) - 1) and check_line:
                            logging.error('mutation position out of protein length range in line %d, annotation %s' % (counter, line[2]))
                            # sys.exit(1)
                            check_line = False
                            anno_line = 'mutation position out of protein length range in line %d, annotation %s' % (counter, line[2])
                        # check AA match between annotation and FASTA
                        if (ori != line[0][pos - 1 : pos + len(ori) - 1]) and check_line:
                            logging.error('Expected %s in reference protein position %d, but %s in line %d mutation annotation' % (line[0][pos - 1 : pos + len(ori) - 1], pos, line[2], counter))
                            # sys.exit(1)
                            check_line = False
                            anno_line = 'Expected %s in reference protein position %d, but %s in line %d mutation annotation' % (line[0][pos - 1 : pos + len(ori) - 1], pos, line[2], counter)
                    if check_line:
                        p1_list.append(line[0])
                        p2_list.append(line[1])
                        ori_list.append(ori)
                        mut_list.append(mut)
                        pos_list.append(pos)
                        check_list.append(check_line)
                        anno_list.append(anno_line)
                    else:
                        p1_list.append('nan')
                        p2_list.append('nan')
                        ori_list.append('nan')
                        mut_list.append('nan')
                        pos_list.append('nan')
                        check_list.append(check_line)
                        anno_list.append(anno_line)
            return p1_list, p2_list, ori_list, mut_list, pos_list, check_list, anno_list
        
    def cookSeq(self, p1, p2, ori, mut, pos, valid_list):
        # get sequence window
        p1_ori, p1_mut = self.getSeq(p1, pos=pos, ori=ori, mut=mut, valid_list=valid_list)
        p2_seq = self.getSeq(p2, partner=True, valid_list=valid_list)
        print('all sequence encoding feature finished.')

        return p1_ori, p1_mut, p2_seq
    
    def getSeq(self, seq, valid_list, pos=None, ori=None, mut=None, partner=False):
        aadict = {'0':0, 'D':1, 'S':2, 'Q':3, 'K':4,
            'I':5, 'P':6, 'T':7, 'F':8, 'N':9,
            'G':10, 'H':11, 'L':12, 'R':13, 'W':14,
            'A':15, 'V':16, 'E':17, 'Y':18, 'M':19, 
            'C':20}
        if partner:
            seq_list = []
            for i in range(len(seq)):
                if valid_list[i]:
                    seq[i] = seq[i] + '0' * 1024
                    seq_list.append([aadict[x] for x in seq[i][:1024]])
                else:
                    seq_list.append([0])
            seq_list = pad_sequences(seq_list, maxlen=1024, padding='post')
            return seq_list
        else:
            ori_list = []
            mut_list = []
            for i in range(len(seq)):
                if valid_list[i]:
                    ori_tmp = '0' * 26 + seq[i] + '0' * 26
                    ori_list.append([aadict[x] for x in ori_tmp[pos[i] : pos[i] + 51]])
                    mut_tmp = ori_tmp[0 : pos[i] + 25] + mut[i] + ori_tmp[pos[i] + 25 + len(ori[i]):]
                    mut_list.append([aadict[x] for x in mut_tmp[pos[i] : pos[i] + 51]])
                else:
                    ori_list.append([0])
                    mut_list.append([0])
            ori_list = pad_sequences(ori_list, maxlen=51, padding='post')
            mut_list = pad_sequences(mut_list, maxlen=51, padding='post')
            return ori_list, mut_list
    

    def getPssm(self, seq, id, valid_list, pos=None, partner=False):
        if valid_list:
            fasta_path = os.path.join(self.tmp_path, id + '.fasta')
            blast_path = os.path.join(self.tmp_path, id + '.blast')
            pssm_path = os.path.join(self.tmp_path, id + '.pssm')
            log_path = os.path.join(self.tmp_path, id + '.log')
            with open(fasta_path, 'w') as f:
                f.write('>' + id + '\n')
                f.write(seq)
            
            cmd = '%s -db %s -query %s -out %s -inclusion_ethresh 0.001 -out_ascii_pssm %s -num_iterations 3 -num_threads 8' % (self.psiblast_bin, self.db_path, fasta_path, blast_path, pssm_path)
            with open(log_path, 'w') as f:
                subprocess.run(cmd, stderr=subprocess.STDOUT, stdout=f)

            lines = 'nan'
            try:
                with open(pssm_path, 'r') as f:
                    lines = f.readlines()
            except IOError as e:
                print(e)
                print('Can not open file: %s, which mean %s did not successfully generate PSSM' % (pssm_path, fasta_path))
                # lines = 'nan'
                return 'nan'
            
            pssmvalue = np.array([])
            for line in lines:
                if len(line.split()) == 44:
                    pssmvalue = np.r_[pssmvalue, np.array(line.split()[2:22]).astype(float)]
            pssmvalue = np.reshape(pssmvalue, (-1, 20))
            if partner:
                pssmvalue = np.r_[pssmvalue, np.zeros([1024, 20])]
                pssmvalue = pssmvalue[:1024, :]
            else:
                pssmvalue = np.r_[np.zeros([26, 20]), pssmvalue, np.zeros([26, 20])]
                pssmvalue = pssmvalue[pos: pos + 51, :]
            
            return pssmvalue
        else:
            return 'nan'
        

    def cookData(self, p1_ori, p1_mut, p2_seq, p1_ori_pssm, p1_mut_pssm, p2_pssm, valid_list, anno):
        # bad_items = np.zeros(len(p1_ori))
        p1_ori_pssm_ = []
        p1_mut_pssm_ = []
        p2_pssm_ = []
        for i in range(len(p1_ori)):
            if ((p1_ori_pssm[i] == 'nan') or (p1_mut_pssm[i] == 'nan') or (p2_pssm[i] == 'nan')):
                if valid_list[i]:
                    valid_list[i] = False
                    anno[i] = 'PSIBLAST unable to generate PSSM for this line'
            else:
                p1_ori_pssm_.append(p1_ori_pssm[i])
                p1_mut_pssm_.append(p1_mut_pssm[i])
                p2_pssm_.append(p2_pssm[i])
        valid_list = np.array(valid_list)
        p1_ori = p1_ori[np.where(valid_list)]
        p1_mut = p1_mut[np.where(valid_list)]
        p2_seq = p2_seq[np.where(valid_list)]
        p1_ori_pssm = np.array(p1_ori_pssm_)
        p1_mut_pssm = np.array(p1_mut_pssm_)
        p2_pssm = np.array(p2_pssm_)

        return p1_ori, p1_mut, p2_seq, p1_ori_pssm, p1_mut_pssm, p2_pssm, valid_list, anno
    
    def outputFile(self, input_path, output_path, pred_class=None, pred_score=None, valid_list=None, anno=None):
        pred_class_dic = {0:'disrupting', 1:'decreasing', 2:'no effect', 3:'increasing'}
        with open(input_path, 'r') as fin:
            with open(output_path, 'w') as fout:
                counter = 0
                valid_counter = 0
                fout.write('affected protein\tpartner protein\tvariant\tpred class\tpred score\terror msg\n')
                for line in fin:
                    if valid_list[counter]:
                        fout.write('%s\t%s\t%f\t.\n' % (line.strip(), pred_class_dic[pred_class[valid_counter]], pred_score[valid_counter]))
                        valid_counter += 1
                    else:
                        fout.write('%s\tNA\tNA\t%s\n' % (line.strip(), anno[counter]))
                    counter += 1
        print('output file finished in %s' % output_path)

        


        


        






