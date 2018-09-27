#!/usr/bin/env python
# encoding: utf-8

import sys
import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

# Raw formats:
# 1. true: all indices and values, predicted: all indices and values
# Should change to:
#    true: query indexed lists (sequentially ordered), predicted: query indexed lists (sequentially ordered)

def dcg_score(y_true, y_score, k=10):
    """Discounted cumulative gain (DCG) at rank k

    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).

    y_score : array-like, shape = [n_samples]
        Predicted scores.

    k : int
        Rank.

    Returns
    -------
    DCG @k : float
    """
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])

    gains = 2 ** y_true - 1

    # highest rank is 1 so +2 instead of +1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)

def ndcg_score(y_true, y_score, k=10):
    """Normalized discounted cumulative gain (NDCG) at rank k

    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).

    y_score : array-like, shape = [n_samples]
        Predicted scores.

    k : int
        Rank.

    Returns
    -------
    NDCG @k : float
    """
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best



def average_precision(y_true, y_score, k=None):
    """Average precision at rank k

    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).

    y_score : array-like, shape = [n_samples]
        Predicted scores.

    k : int
        Rank.

    Returns
    -------
    average precision @k : float
    """
    assert len(y_true) == len(y_score)
    unique_y = np.unique(y_true)

    if len(unique_y) > 2:
        raise ValueError("Only supported for two relevance levels.")
    elif len(unique_y) <= 1:
        return 0.0

    if k is None:
        k = len(y_true)
    else:
        k = min(k, len(y_true))

    pos_label = unique_y[1]
    n_pos = np.sum(y_true == pos_label)

    order = np.argsort(y_score)[::-1][:k]
    y_true = np.asarray(y_true)[order]

    score = 0
    cur_count = 0
    for i in xrange(len(y_true)):
        if y_true[i] == pos_label:
            # Compute precision up to document i
            # i.e, percentage of relevant documents up to document i.
            cur_count += 1
            score += cur_count / (i + 1.0)

    if cur_count == 0:
        return 0

    return score / cur_count

# preprocessing for reading data, return a dict of sorted list (of the predicting data)
# rank it in according to index
def read_score_file(filename):
    score_lists = {}
    for line in open(filename):
        pair = line.split()
        q = pair[0]
        doc = pair[1]
        score = float(pair[2])
        if q not in score_lists:
            score_lists[q] = [[],[]]
        score_lists[q][0].append(doc)
        score_lists[q][1].append(score)
    index_ranked_lists = {}
    for q in score_lists:
        docs, scores = zip(*sorted(zip(score_lists[q][0],score_lists[q][1])))
        index_ranked_lists[q] = list(scores)
    return index_ranked_lists

# remove queries that have no relevant documents
def strip_all_0_queries(y_dict):
    y_stripped_dict = {}
    for q in y_dict:
        val = y_dict[q]
        if np.sum(val) != 0:
            y_stripped_dict[q] = val
    return y_stripped_dict

def eval_RMSE(Y_true, Y_pred, x_index, y_index):
    return np.sqrt(np.mean((np.asarray(Y_true[x_index, y_index]).flatten() - Y_pred[x_index, y_index])**2))

def eval_MAP(relevanceLists, predicted):
    aps = []
    for q in relevanceLists:
        lhat = predicted[q]
        ltruth = relevanceLists[q]
        correct = 0.0
        sum_precision = 0.0
        for pos in range(len(lhat)):
            if lhat[pos] in ltruth:
                correct = correct + 1
                sum_precision = sum_precision + correct/(pos+1)
        aps.append(sum_precision/len(ltruth))
    return sum(aps)/len(aps)

# not including MAP and AUC
def eval_cf_scores(y_true_dict, y_score_dict):
    # when calculating RMSE, make sure y_true_dict is the full dict of list
    ndcgs = [[], [], []] # return ndcg at 1, 3, 5
    for q in y_true_dict:
        if q not in y_score_dict:
            raise ValueError("Prediction has missing items.")
        if np.sum(y_true_dict[q]) != 0:
            ndcgs[0].append(ndcg_score(y_true_dict[q], y_score_dict[q], k=1))
            ndcgs[1].append(ndcg_score(y_true_dict[q], y_score_dict[q], k=3))
            ndcgs[2].append(ndcg_score(y_true_dict[q], y_score_dict[q], k=5))
    ndcgs = np.asarray(ndcgs)

    y_true_list = trans_dict_to_list(y_true_dict, y_true_dict)
    y_score_list = trans_dict_to_list(y_true_dict, y_score_dict)

    rmse = np.mean((y_true_list - y_score_list)**2)

    # ndcg@1, ndcg@3, ndcg@5, rmse
    return np.mean(ndcgs[0,:]), np.mean(ndcgs[1,:]), np.mean(ndcgs[2,:]), np.sqrt(rmse)

# including MAP and AUC
def eval_all_scores(y_true_dict, y_score_dict):
    # when calculating RMSE, make sure y_true_dict is the full dict of list
    aps = []    # average precisions
    ndcgs = [[], [], []] # return ndcg at 1, 3, 5
    for q in y_true_dict:
        if q not in y_score_dict:
            raise ValueError("Prediction has missing items.")
        if np.sum(y_true_dict[q]) != 0:
            aps.append(average_precision(y_true_dict[q], y_score_dict[q]))
            ndcgs[0].append(ndcg_score(y_true_dict[q], y_score_dict[q], k=1))
            ndcgs[1].append(ndcg_score(y_true_dict[q], y_score_dict[q], k=3))
            ndcgs[2].append(ndcg_score(y_true_dict[q], y_score_dict[q], k=5))
    ndcgs = np.asarray(ndcgs)

    y_true_list = trans_dict_to_list(y_true_dict, y_true_dict)
    y_score_list = trans_dict_to_list(y_true_dict, y_score_dict)

    auc = roc_auc_score(y_true_list, y_score_list)
    rmse = np.mean((y_true_list - y_score_list)**2)

    # map, ndcg@1, ndcg@3, ndcg@5, auc, rmse
    return sum(aps)/len(aps), np.mean(ndcgs[0,:]), np.mean(ndcgs[1,:]), np.mean(ndcgs[2,:]), auc, np.sqrt(rmse)

# including MAP and AUC
def eval_all_scores_from_array(y_true_array, y_score_array, mask):
    y_true_dict = trans_array_to_dict(y_true_array, mask)
    y_score_dict = trans_array_to_dict(y_score_array, mask)
    return eval_all_scores(y_true_dict, y_score_dict)

# excluding MAP and AUC
def eval_cf_scores_from_array(y_true_array, y_score_array, mask):
    y_true_dict = trans_array_to_dict(y_true_array, mask)
    y_score_dict = trans_array_to_dict(y_score_array, mask)
    return eval_cf_scores(y_true_dict, y_score_dict)

# return the dict, note 'mask' has to be of dtype boolean
def trans_array_to_dict(array, mask):
    ret_dict = {}
    for i in xrange(mask.shape[0]):
        ret_dict[i] = array[i][mask[i]]
    return ret_dict


# return the numpy array list
def trans_dict_to_list(keys, y_dict):
    stack_list = [y_dict[key] for key in keys]
    return np.hstack(stack_list)

def eval_all_scores_from_file(y_true_file, y_score_file):
    y_true_dict = read_score_file(y_true_file)
    y_score_dict = read_score_file(y_score_file)
    return eval_all_scores(y_true_dict, y_score_dict)

def eval_cf_scores_from_file(y_true_file, y_score_file):
    y_true_dict = read_score_file(y_true_file)
    y_score_dict = read_score_file(y_score_file)
    return eval_cf_scores(y_true_dict, y_score_dict)


if __name__ == '__main__':
    y_true_dict = read_score_file('/clair/yuexinw/research/conv_cf/relational/split_data/cmu/link.tes.5.txt')
    y_pred_dict = read_score_file('/clair/yuexinw/research/conv_cf/TOP-plus-plus/split_cmu_pred_dir/prediction')
    # y_true_dict = strip_all_0_queries(y_true_dict)
    print eval_all_scores(y_true_dict, y_pred_dict)