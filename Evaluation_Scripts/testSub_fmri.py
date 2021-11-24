#!/usr/bin/env python
# This script computes the score for the comparison of Model RDM with
# the fMRI data 
# Note: Remember to use the appropriate noise ceiling correlation values for the dataset you are testing
# e.g. nc118_EVC_R2 for the 118-image training set.

import os
import sys
import argparse
import glob
import json

import h5py
import numpy as np
from scipy import stats
from scipy.spatial.distance import squareform
from scipy import io

#defines the noise ceiling squared correlation values for EVC and IT, for the training (92, 118) and test (78) image sets 
nc92_EVC_R2 = 0.1589
nc92_IT_R2 = 0.3075
nc92_avg_R2 = (nc92_EVC_R2+nc92_IT_R2)/2.

nc118_EVC_R2 = 0.1048
nc118_IT_R2 = 0.0728
nc118_avg_R2 = (nc118_EVC_R2+nc118_IT_R2)/2.

nc78_EVC_R2 = 0.0640
nc78_IT_R2 = 0.0647
nc78_avg_R2 = (nc78_EVC_R2+nc78_IT_R2)/2.


#loads the input files if in .mat format
def loadmat(matfile):
    try:
        f = h5py.File(matfile, 'r')
    except (IOError, OSError):
        return io.loadmat(matfile)
    else:
        return {name: np.transpose(f.get(name)) for name in f.keys()}


def loadnpy(npyfile):
    return np.load(npyfile)


def load(data_file):
    root, ext = os.path.splitext(data_file)
    return {'.npy': loadnpy,
            '.mat': loadmat
            }.get(ext, loadnpy)(data_file)


def sq(x):
    return squareform(x, force='tovector', checks=False)


#defines the spearman correlation
def spearman(model_rdm, rdms):
    model_rdm_sq = sq(model_rdm)
    return [stats.spearmanr(sq(rdm), model_rdm_sq)[0] for rdm in rdms]


#computes spearman correlation (R) and R^2, and ttest for p-value.
def fmri_rdm(model_rdm, fmri_rdms):
    corr = spearman(model_rdm, fmri_rdms)
    corr_squared = np.square(corr)
    return np.mean(corr_squared), stats.ttest_1samp(corr_squared, 0)[1]


def evaluate(submission, targets, target_names=['EVC_RDMs', 'IT_RDMs']):
    out = {name: fmri_rdm(submission[name], targets[name]) for name in target_names}
    out['score'] = np.mean([x[0] for x in out.values()])
    return out


#function that evaluates the RDM comparison.    
def test_fmri_submission(submit_file):
    score = {}
    target_file = 'target_fmri.mat'
    target = load(target_file)
    submit = load(submit_file)
    out = evaluate(submit, target)
    score['EVC'] = ((out['EVC_RDMs'][0])/nc92_EVC_R2)*100.      #evc percent of noise ceiling
    score['IT'] = ((out['IT_RDMs'][0])/nc92_IT_R2)*100.         #it percent of noise ceiling
    score['Avg'] = ((out['score'])/nc92_avg_R2)*100.      #avg (score) percent of noise ceiling
    return score


def main():
    parser = argparse.ArgumentParser(description='calculate R-squared score')
    parser.add_argument('-rd','--rdms_dir', help='rdms directory path', default = "../Feature_Extract/rdms/92images_rdms", type=str)
    args = vars(parser.parse_args())
    net_dir = os.path.join(args['rdms_dir'])
    net_dir_list = glob.glob(net_dir+'/*')
    net_list = [x.split('\\')[-1] for x in net_dir_list]
    
    score = {}
    for net in net_list:
        score_ = {}
        layer_dir = net_dir + '/' + net + '/pearson'
        layer_dir_list = glob.glob(layer_dir+'/*')
        layer_list = [x.split('\\')[-1] for x in layer_dir_list]
        if 'args.json' in layer_list:
            layer_list.remove('args.json')
        for layer in layer_list:
            submit_file = layer_dir + '/' + layer + '/submit_fmri.mat'
            score_[layer] = test_fmri_submission(submit_file)
        score[net] = score_

    with open('score.json', 'w') as fp:
        json.dump(score, fp, sort_keys=True, indent=4)
    
    
if __name__ == '__main__':
    main()