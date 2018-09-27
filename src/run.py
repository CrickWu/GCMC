import numpy as np
import scipy.sparse as sp
import sys, os
from joblib import Parallel, delayed
import logging
import argparse

from BGC_model import BGC_model
from utils.eval import eval_cf_scores_from_array, eval_all_scores_from_array, eval_MAP, eval_RMSE # evaluation
from utils.data_parse import load_sp_graph, dump_Y_pred_to_ranked_lists, get_relevance_lists # data manipulation

def run_single_model(args, fold=1):
    """
    return: the scores lists under the best validation result
    """

    dataset = args.dataset
    pos_up_ratio = args.pos_up_ratio
    k = args.layers[0]  #  the hidden dimension (SVD for the training matrix)
    hidden_list = args.layers[1:]
    iter = args.iters
    save_dir = args.save_dir
    seed= args.seed

    filename = save_dir + dataset + '.%d.' %fold +'k%d.' %(k) + '_'.join(map(str, hidden_list)) \
               + '.pos_up_ratio%.1f' %(pos_up_ratio)
    logger = logging.getLogger('%s fold %d' %(dataset, fold))
    logger.setLevel(logging.INFO)
    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')

    if args.save_log:
        fh = logging.FileHandler(filename + '.log', 'w')
        fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info("running fold %d, k %d, hidden_list %r, pos_up_ratio %.1f" % (fold, k, hidden_list, pos_up_ratio))

    data_path = args.data_dir + '%s/' %(dataset)
    trn_file = data_path + 'link.trn.%d.txt' %(fold)
    val_file = data_path + 'link.val.%d.txt' %(fold)
    tst_file = data_path + 'link.tes.%d.txt' %(fold)
    g, _, _ = load_sp_graph(data_path + 'g.graph')
    h, _, _ = load_sp_graph(data_path + 'h.graph')
    trn_graph, trn_x_index, trn_y_index = load_sp_graph(trn_file, shape=(g.shape[0], h.shape[0]))
    val_graph, val_x_index, val_y_index = load_sp_graph(val_file, shape=(g.shape[0], h.shape[0]))
    tst_graph, tst_x_index, tst_y_index = load_sp_graph(tst_file, shape=(g.shape[0], h.shape[0]))

    val_mask = sp.csr_matrix(([1] * len(val_x_index), (val_x_index, val_y_index)), shape=(g.shape[0], h.shape[0]), dtype=bool).toarray()
    tst_mask = sp.csr_matrix(([1] * len(tst_x_index), (tst_x_index, tst_y_index)), shape=(g.shape[0], h.shape[0]), dtype=bool).toarray()
    val_graph_dense = val_graph.toarray()
    tst_graph_dense = tst_graph.toarray()
    tst_relevance_lists = get_relevance_lists(tst_file)
    val_relevance_lists = get_relevance_lists(val_file)
    trn_relevance_lists = get_relevance_lists(trn_file)
    lr = 0.1

    logger.info('data loaded')

    model = BGC_model(lr=lr, hidden_list=hidden_list, seed=seed, model_file=filename + '.param')

    model.add_data(g, h, trn_graph, trn_x_index, trn_y_index, tst_graph, tst_x_index, tst_y_index,
                   k=k, pos_up_ratio=pos_up_ratio) # here comes the svd
    logger.info('data added')

    model.build(pre_load=False, binary_graph=args.binary_graph) # here comes the compiling procedure
    logger.info('model built')

    if args.binary_graph:
        # MAP criterion
        opt_tst_score = 0.0
        opt_val_score = 0.0

        opt_val_list = []  # iter, trn_map, val_map, tst_map
        opt_tst_list = []
        opt_val_val_score_list = [] # map, ndcg@1, ndcg@3, ndcg@5, auc, rmse
        opt_val_tst_score_list = [] # map, ndcg@1, ndcg@3, ndcg@5, auc, rmse

        for i in xrange(iter):
            Y_pred, loss, grad_norm = model.step_train(max_iter = 1)
            if i % 10 == 0:
                trn_map = eval_MAP(trn_relevance_lists, dump_Y_pred_to_ranked_lists(Y_pred, trn_x_index, trn_y_index))
                val_map = eval_MAP(val_relevance_lists, dump_Y_pred_to_ranked_lists(Y_pred, val_x_index, val_y_index))
                tst_map = eval_MAP(tst_relevance_lists, dump_Y_pred_to_ranked_lists(Y_pred, tst_x_index, tst_y_index))

                if val_map > opt_val_score:
                    if args.save_model:
                        model.store_params()
                    opt_val_list = [i, trn_map, val_map, tst_map]
                    opt_val_val_score_list = eval_all_scores_from_array(val_graph_dense, Y_pred, val_mask)
                    opt_val_tst_score_list = eval_all_scores_from_array(tst_graph_dense, Y_pred, tst_mask)
                if tst_map > opt_tst_score:
                    opt_tst_list = [i, trn_map, val_map, tst_map]

                opt_val_score = val_map > opt_val_score and val_map or opt_val_score
                opt_tst_score = tst_map > opt_tst_score and tst_map or opt_tst_score
                logger.info('%d: trn MAP: %.4f val MAP: %.4f max val MAP: %.4f tst MAP: %.4f max tst MAP: %.4f' %(i, trn_map, val_map, opt_val_score, tst_map, opt_tst_score))

        logger.info('opt_tst_iter: {:d}, trn: {:.4f}, val: {:.4f}, tst {:.4f}'.format(*opt_tst_list))
        logger.info('opt_val_iter: {:d}, trn: {:.4f}, val: {:.4f}, tst {:.4f}'.format(*opt_val_list))
        logger.info('opt_val val scores: map {:.4f} ndcg@1: {:.4f}, ndcg@3: {:.4f}, ndcg@5 {:.4f}, auc {:.4f}, rmse {:.4f}'.format(*opt_val_val_score_list))
        logger.info('opt_val tst scores: map {:.4f} ndcg@1: {:.4f}, ndcg@3: {:.4f}, ndcg@5 {:.4f}, auc {:.4f}, rmse {:.4f}'.format(*opt_val_tst_score_list))
    else:
        # RMSE criterion
        opt_tst_score = 1e10
        opt_val_score = 1e10

        opt_val_list = []  # iter, trn_map, val_map, tst_map
        opt_tst_list = []

        opt_val_val_score_list = [] # ndcg@1, ndcg@3, ndcg@5, rmse
        opt_val_tst_score_list = [] # ndcg@1, ndcg@3, ndcg@5, rmse
        for i in xrange(iter):
            Y_pred, loss, grad_norm = model.step_train(max_iter = 1)
            if i % 10 == 0:
                trn_rmse = eval_RMSE(trn_graph, Y_pred, trn_x_index, trn_y_index)
                val_rmse = eval_RMSE(val_graph, Y_pred, val_x_index, val_y_index)
                tst_rmse = eval_RMSE(tst_graph, Y_pred, tst_x_index, tst_y_index)

                if val_rmse < opt_val_score:
                    model.store_params()
                    opt_val_list = [i, trn_rmse, val_rmse, tst_rmse]
                    opt_val_val_score_list = eval_cf_scores_from_array(val_graph_dense, Y_pred, val_mask)
                    opt_val_tst_score_list = eval_cf_scores_from_array(tst_graph_dense, Y_pred, tst_mask)
                if tst_rmse < opt_tst_score:
                    opt_tst_list = [i, trn_rmse, val_rmse, tst_rmse]

                opt_val_score = val_rmse < opt_val_score and val_rmse or opt_val_score
                opt_tst_score = tst_rmse < opt_tst_score and tst_rmse or opt_tst_score
                logger.info('%d: trn RMSE: %.4f val RMSE: %.4f min val RMSE: %.4f tst RMSE: %.4f min tst RMSE: %.4f' %(i, trn_rmse, val_rmse, opt_val_score, tst_rmse, opt_tst_score))

        logger.info('opt_val_iter: {:d}, trn: {:.4f}, val: {:.4f}, tst {:.4f}'.format(*opt_val_list))
        logger.info('opt_tst_iter: {:d}, trn: {:.4f}, val: {:.4f}, tst {:.4f}'.format(*opt_tst_list))
        logger.info('opt_val val scores: ndcg@1: {:.4f}, ndcg@3: {:.4f}, ndcg@5 {:.4f}, rmse {:.4f}'.format(*opt_val_val_score_list))
        logger.info('opt_val tst scores: ndcg@1: {:.4f}, ndcg@3: {:.4f}, ndcg@5 {:.4f}, rmse {:.4f}'.format(*opt_val_tst_score_list))

    logger.removeHandler(fh)
    if args.save_log:
        logger.removeHandler(ch)

    return opt_val_val_score_list, opt_val_tst_score_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_jobs', type=int, default=1, help='parallel job, better = #cpu_threads')
    parser.add_argument('--pos_up_ratio', type=float, default=10., help='postive edge weight ratio, useful to encourage positive edge prediction in binary tasks (`binary_graph`)')
    parser.add_argument('--layers', nargs='+', type=int, default=[80, 70, 5], help='feedforward network structure')
    parser.add_argument('--iters', type=int, default=1000, help='number of iterations')
    parser.add_argument('--dataset', type=str, default='cmu', help='dataset')
    parser.add_argument('--binary_graph', action='store_true', help='whether the edge values are binary')
    parser.add_argument('--fold', type=int, default=1, help='number of folds to evaluate (max 5)')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    # miscs
    parser.add_argument('--summary_name', type=str, default='gcmc.log', help='meta-log summary filename')
    parser.add_argument('--save_model', action='store_true', help='whether to save the best model in option `save_dir`')
    parser.add_argument('--save_log', action='store_true', help='whether to save the intermediate logs in option `save_dir`')
    parser.add_argument('--data_dir', type=str, default='data/', help='directory for reading input data')
    parser.add_argument('--save_dir', type=str, default='save/', help='directory for saving models and parameters')

    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    # main logger
    logger = logging.getLogger('gcmc')
    logger.setLevel(logging.INFO)
    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')

    fh = logging.FileHandler(args.summary_name, 'w')
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    res = Parallel(n_jobs=args.n_jobs)(delayed(run_single_model) (args=args, fold=i+1) for i in xrange(args.fold))

    res = np.asarray(res)
    logger.info("%s, layers %r, pos_up_ratio %.1f, #fold %d" % (args.dataset, args.layers, args.pos_up_ratio, args.fold))

    if args.binary_graph:
        logger.info('opt_val val scores: map {:.4f} ndcg@1: {:.4f}, ndcg@3: {:.4f}, ndcg@5 {:.4f}, auc {:.4f}, rmse {:.4f}'.format(*res[:,0,:].mean(axis=0)))
        logger.info('opt_val tst scores: map {:.4f} ndcg@1: {:.4f}, ndcg@3: {:.4f}, ndcg@5 {:.4f}, auc {:.4f}, rmse {:.4f}'.format(*res[:,1,:].mean(axis=0)))
    else:
        logger.info('opt_val val scores, ndcg@1: {:.4f}, ndcg@3: {:.4f}, ndcg@5 {:.4f}, rmse {:.4f}'.format(*res[:,0,:].mean(axis=0)))
        logger.info('opt_val tst scores, ndcg@1: {:.4f}, ndcg@3: {:.4f}, ndcg@5 {:.4f}, rmse {:.4f}'.format(*res[:,1,:].mean(axis=0)))