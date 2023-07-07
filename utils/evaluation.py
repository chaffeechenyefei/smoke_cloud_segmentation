import numpy as np


def print_tpr_wiz_fpr(y_true,y_score,fpr=[1e-3,1e-4]):
    Npos = y_true.sum()
    Nneg = len(y_true) - Npos
    print('#N = %d, #P = %d'%(Nneg,Npos))
    pos_array = y_score[y_true == 1]
    neg_array = y_score[y_true == 0]

    print('sorting for neg array')
    neg_array.sort()
    neg_array = neg_array[::-1]

    for _fpr in fpr:
        n_fp = int(_fpr * Nneg)
        threshold = neg_array[n_fp:n_fp + 2].mean()
        n_tp = np.array(pos_array >= threshold, np.int32).sum()
        tpr = n_tp / Npos
        print('fpr = %0.5f, tpr = %0.5f, threshold = %0.3f' % (_fpr, tpr, threshold))


def print_precision_wiz_recall(y_true,y_score,recall=[0.9,0.8,0.7,0.6,0.5]):
    Npos = y_true.sum()
    Nneg = len(y_true) - Npos
    print('#N = %d, #P = %d'%(Nneg,Npos))
    pos_array = y_score[y_true == 1]
    neg_array = y_score[y_true == 0]

    print('sorting for pos array')
    pos_array.sort()
    pos_array = pos_array[::-1]

    #recall = tpr
    for _tpr in recall:
        n_tp = int(_tpr * Npos)
        threshold = pos_array[n_tp:n_tp + 2].mean()

        n_fp = np.array(neg_array >= threshold, np.int32).sum()
        precision = n_tp / (n_tp + n_fp)
        print('recall = %0.5f, precision = %0.5f, threshold = %0.3f' % (_tpr, precision, threshold))