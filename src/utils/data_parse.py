import numpy as np
import scipy.sparse as sp

# minus each index by 1
def load_sp_graph(filename, shape=None):
    data = []
    row = []
    col = []
    for line in open(filename):
        fields = line.strip().split()
        data.append(float(fields[2]))
        row.append(int(fields[0]) - 1)
        col.append(int(fields[1]) - 1)
    if shape is None:
        return sp.csr_matrix((data, (row, col)), dtype='float32'), row, col
    else:
        return sp.csr_matrix((data, (row, col)), shape=shape, dtype='float32'), np.asarray(row), np.asarray(col)


# note the indices have to be shifted by 1
def dump_Y_pred_to_ranked_lists(Y_pred, x_index, y_index):
    scoreLists = {}
    rankedLists = {}
    for i, j in zip(x_index, y_index):
        if (i+1) not in scoreLists:
            scoreLists[i+1] = [[], []]
        scoreLists[i+1][0].append(j+1)
        scoreLists[i+1][1].append(Y_pred[i, j])
    for q in scoreLists:
        scores, docs = zip(*sorted(zip(scoreLists[q][1],scoreLists[q][0]), reverse=True))
        rankedLists[q] = docs
    return rankedLists

# read the groud-truth link data from file
def get_relevance_lists(fileName):
    relevanceLists = {}
    for line in open(fileName):
        pair = line.split()
        q = int(pair[0])
        if int(pair[2])== 1:
            if q not in relevanceLists:
                relevanceLists[q] = []
            relevanceLists[q].append(int(pair[1]))
    return relevanceLists