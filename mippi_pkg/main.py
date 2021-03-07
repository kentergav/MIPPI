#!/usr/bin/env python
# coding: utf-8

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import sys
import argparse
import pathlib
import shutil
import multiprocessing
import itertools
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

from utils import utils
from NetBuild import *


def main():
    # input command
    parser = argparse.ArgumentParser(description='Mutation Impact on Protein-Protein Interaction')
    parser.add_argument('-i',dest='inputfile' , type=pathlib.Path, help='input interaction table path', required=True)
    parser.add_argument('-o',dest='outputfile' , type=pathlib.Path, help='output file path', required=True)
    parser.add_argument('-bin',dest='bin' , type=pathlib.Path, help='psiblast bin folder path, ${bin/psiblast}', required=True)
    parser.add_argument('-db',dest='db' , type=pathlib.Path, help='psiblast search database path, recommand UNIREF90', required=True)
    parser.add_argument('-id',dest='id', action='store_true', help='with -id, uniprot ID in inputfile, instead of fasta')
    parser.add_argument('-notmp',dest='notmp', action='store_true', help='with -notmp, delete all intermediate file in /tmp folder, including fasta, PSSM and blast output file')
    
    args = parser.parse_args()

    # check format
    util = utils(psiblast_bin=args.bin, db_path=args.db, tmp_path='./mippi_tmp')

    if args.id:
        p1, p2, var = util.splitItems(args.inputfile)
        print('Start acquiring FASTA from UNIPROT.org ...')
        with multiprocessing.Pool() as p:
            p1 = p.map(util.getFasta, p1)
            print('Acquired affected proteins fasta (reference) from UNIPROT')
        with multiprocessing.Pool() as p:
            p2 = p.map(util.getFasta, p2)
            print('Acquired partner proteins fasta (reference) from UNIPROT')
        trans_file = util.transId2FastaFile(p1, p2, var)
        p1, p2, ori, mut, pos, check, anno = util.checkFormat(trans_file)
    else:
        p1, p2, ori, mut, pos, check, anno = util.checkFormat(args.inputfile)


    # # -----------------------------------------------------for debug input ------------------------------------------------------
    # psiblast_bin=r'D:\SoftWare\blast\blast-2.11.0+\bin\psiblast'
    # db_path=r'D:\SoftWare\blast\db\swissprot'
    # inputfile = 'example2.txt'
    # outputfile = 'results2.txt'
    # id_ = True
    # notmp = True

    # util = utils(psiblast_bin=psiblast_bin, db_path=db_path, tmp_path='./mippi_tmp')

    # if id_:
    #     p1, p2, var = util.splitItems(inputfile)
    #     print('Start acquiring FASTA from UNIPROT.org ...')
    #     with multiprocessing.Pool() as p:
    #         p1 = p.map(util.getFasta, p1)
    #         print('Acquired affected proteins fasta (reference) from UNIPROT')
    #     with multiprocessing.Pool() as p:
    #         p2 = p.map(util.getFasta, p2)
    #         print('Acquired partner proteins fasta (reference) from UNIPROT')
    #     trans_file = util.transId2FastaFile(p1, p2, var)
    #     p1, p2, ori, mut, pos, check, anno = util.checkFormat(trans_file)
    # else:
    #     p1, p2, ori, mut, pos, check, anno = util.checkFormat(inputfile)
    # # -----------------------------------------------------------------------------------------------------------------------------


    
    

    p1_ori, p1_mut, p2_seq = util.cookSeq(p1, p2, ori, mut, pos, check)

    # get pssm feature
    ## first get whole sequences
    p1_mut_whole = []
    for i in range(len(p1)):
        if check[i]:
            p1_mut_whole.append(p1[i][: pos[i] - 1] + mut[i] + p1[i][pos[i] - 1 + len(ori[i]):])
        else:
            p1_mut_whole.append([0])
    
    # get PSSM
    print('Start generate PSSM using PSIBLAST on %d threads, this step may take long time ...' % (multiprocessing.cpu_count()))
    with multiprocessing.Pool() as p:
        p1_ori_pssm = p.starmap(util.getPssm, zip(p1, ['p1_ori_whole_line' + str(x) for x in range(len(p1))], check, pos))
        print('affected proteins (reference) PSSM finished')
    with multiprocessing.Pool() as p:
        p1_mut_pssm = p.starmap(util.getPssm, zip(p1_mut_whole, ['p1_mut_whole_line' + str(x) for x in range(len(p1_mut_whole))], check, pos))
        print('affected proteins (mutant) PSSM finished')
    with multiprocessing.Pool() as p:
        p2_pssm = p.starmap(util.getPssm, zip(p2, ['p2_seq_line' + str(x) for x in range(len(p2))], check, itertools.repeat(None), itertools.repeat(True)))
        print('partner proteins PSSM finished')
    
    # cook all the features
    p1_ori, p1_mut, p2_seq, p1_ori_pssm, p1_mut_pssm, p2_pssm, valid_list, anno = util.cookData(p1_ori, p1_mut, p2_seq, p1_ori_pssm, p1_mut_pssm, p2_pssm, check, anno)

    if len(p1_ori) > 0:
        data = [p1_ori, p1_mut, p2_seq, p1_ori_pssm, p1_mut_pssm, p2_pssm]
        model = build_model()
        adam = Adam(learning_rate=0.0002)
        model.compile(adam, loss=categorical_focal_loss(alpha=[.25, .25, .1, .25], gamma=2.), 
                    metrics=['acc', tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='top2acc')])
        # if have valid items, make prediction
        print('Start making predictions for 5 folds')
        for i in tqdm(range(5)):
            model_path = r'./saved_model/bestAcc.h5' + str(i)
            model.load_weights(model_path).expect_partial()
            pred = model.predict(data)
            if i == 0:
                score = pred
            else:
                score += pred
        
        pred_class = score.argmax(axis=-1)
        pred_score = score.max(axis=-1) / 5

        util.outputFile(args.inputfile, args.outputfile, pred_class, pred_score, valid_list, anno)
        # debug output file------------------------------
        # util.outputFile(inputfile, outputfile, pred_class, pred_score, valid_list, anno)
    else:
        util.outputFile(args.inputfile, args.outputfile, None, None, valid_list, anno)
        # debug output file------------------------------
        # util.outputFile(inputfile, outputfile, None, None, valid_list, anno)
    
    if args.notmp:
        shutil.rmtree('./mippi_tmp')
    
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()

