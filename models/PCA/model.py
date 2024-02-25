import torch
import torch.nn as nn

from src.lap_solvers.sinkhorn import Sinkhorn
from src.feature_align import feature_align
from src.gconv import Siamese_Gconv
from models.PCA.affinity_layer import Affinity
from src.lap_solvers.hungarian import hungarian

from src.utils.config import cfg
from models.PCA.model_config import model_cfg
import my_ops
from my_ops import IPF, my_IPF
from src.backbone import *

is_cuda = torch.cuda.is_available()
def to_var(x):
    if is_cuda:
        x = x.cuda()
    return x
CNN = eval(cfg.BACKBONE)

def calc_differences_from_marginals(cont_s, n_rows, row_marginals, n_cols, column_marginals):

    cont_s = cont_s.view(-1, n_rows, n_cols)

    row_absdiffs = torch.abs(torch.sum(cont_s[:,:n_rows,:], dim = 2) - row_marginals)
    col_absdiffs = torch.abs(torch.sum(cont_s[:,:,:n_cols], dim = 1) - column_marginals)

    rows_avg_absdiff = torch.sum(row_absdiffs)/n_rows
    cols_avg_absdiff = torch.sum(col_absdiffs)/n_cols

    avg_diff = torch.sum(row_absdiffs)/n_rows + torch.sum(col_absdiffs)/n_cols

    if torch.count_nonzero(column_marginals) < len(column_marginals) or torch.count_nonzero(row_marginals) < len(row_marginals):
        partial_matching = True
    else:
        partial_matching = False
    with open(cfg.DATASET_NAME +'CUB_newstats_MAE_ptype'+cfg.OPTIMIZATION_METHOD+'_ipf_type'+str(cfg.ipf)+'_src_outlier'+str(cfg.PROBLEM.SRC_OUTLIER)+'_trg_outlier'+str(cfg.PROBLEM.TGT_OUTLIER)+ 'partial_matching_'+str(partial_matching)+'_average_diff_abs_marginals','a') as file_diff:
        file_diff.write("{:.6f}".format(rows_avg_absdiff) +'\t' + str(n_rows.item())+'\t' +"{:.6f}".format(cols_avg_absdiff) +'\t' + str(n_cols.item())+'\n')

def calc_prediction_shift(unnormalized_s, normalized_s):

    pred_unnormalized_s = hungarian(unnormalized_s)
    pred_normalized_s = hungarian(normalized_s)

    m = unnormalized_s.size(dim=0)
    n = unnormalized_s.size(dim=1)

    cardinality = torch.tensor(min(m, n))

    prediction_shift = cardinality - torch.sum(torch.logical_and(pred_unnormalized_s == 1, pred_normalized_s == 1))
    prediction_shift_file = cfg.DATASET_NAME + 'CUB_newstats_prediction_shift_ptype'+cfg.OPTIMIZATION_METHOD+'_ipf_'+str(cfg.ipf)+'.txt'
    with open(prediction_shift_file, 'a') as file_shift:
        file_shift.write(str(m) + '\t' + str(n) + '\t' + str(prediction_shift.item())+ '\n')



def log_sinkhorn_iterations(Z: torch.Tensor, log_mu: torch.Tensor, log_nu: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)

def log_optimal_transport(scores: torch.Tensor, alpha: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m*one).to(scores), (n*one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z


def normalize_over_channels(x):
    channel_norms = torch.norm(x, dim=1, keepdim=True)
    return x / channel_norms

class Net(CNN):
    def __init__(self):
        super(Net, self).__init__()
        self.sinkhorn = Sinkhorn(max_iter=cfg.PCA.SK_ITER_NUM, epsilon=cfg.PCA.SK_EPSILON, tau=cfg.PCA.SK_TAU)
        self.l2norm = nn.LocalResponseNorm(cfg.PCA.FEATURE_CHANNEL * 2, alpha=cfg.PCA.FEATURE_CHANNEL * 2, beta=0.5, k=0)
        self.gnn_layer = cfg.PCA.GNN_LAYER
        #self.pointer_net = PointerNet(cfg.PCA.GNN_FEAT, cfg.PCA.GNN_FEAT // 2, alpha=cfg.PCA.VOTING_ALPHA)
        for i in range(self.gnn_layer):
            if i == 0:
                gnn_layer = Siamese_Gconv(cfg.PCA.FEATURE_CHANNEL * 2, cfg.PCA.GNN_FEAT)
            else:
                gnn_layer = Siamese_Gconv(cfg.PCA.GNN_FEAT, cfg.PCA.GNN_FEAT)
            self.add_module('gnn_layer_{}'.format(i), gnn_layer)
            self.add_module('affinity_{}'.format(i), Affinity(cfg.PCA.GNN_FEAT))
            if i == self.gnn_layer - 2:  # only second last layer will have cross-graph module
                self.add_module('cross_graph_{}'.format(i), nn.Linear(cfg.PCA.GNN_FEAT * 2, cfg.PCA.GNN_FEAT))
        self.cross_iter = cfg.PCA.CROSS_ITER
        self.cross_iter_num = cfg.PCA.CROSS_ITER_NUM
        self.rescale = cfg.PROBLEM.RESCALE
        bin_score = torch.nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)

    def reload_backbone(self):
        self.node_layers, self.edge_layers = self.get_backbone(True)


    def forward(self, data_dict, **kwargs):
        if 'images' in data_dict:
            # real image data
            src, tgt = data_dict['images']
            P_src, P_tgt = data_dict['Ps']
            ns_src, ns_tgt = data_dict['ns']
            A_src, A_tgt = data_dict['As']
            batch_size = data_dict['batch_size']

            # extract feature
            src_node = self.node_layers(src)
            src_edge = self.edge_layers(src_node)
            tgt_node = self.node_layers(tgt)
            tgt_edge = self.edge_layers(tgt_node)

            # feature normalization
            src_node = self.l2norm(src_node)
            src_edge = self.l2norm(src_edge)
            tgt_node = self.l2norm(tgt_node)
            tgt_edge = self.l2norm(tgt_edge)

            # arrange features
            U_src = feature_align(src_node, P_src, ns_src, self.rescale)
            F_src = feature_align(src_edge, P_src, ns_src, self.rescale)
            U_tgt = feature_align(tgt_node, P_tgt, ns_tgt, self.rescale)
            F_tgt = feature_align(tgt_edge, P_tgt, ns_tgt, self.rescale)
        elif 'features' in data_dict:
            # synthetic data
            src, tgt = data_dict['features']
            ns_src, ns_tgt = data_dict['ns']
            A_src, A_tgt = data_dict['As']

            U_src = src[:, :src.shape[1] // 2, :]
            F_src = src[:, src.shape[1] // 2:, :]
            U_tgt = tgt[:, :tgt.shape[1] // 2, :]
            F_tgt = tgt[:, tgt.shape[1] // 2:, :]
        else:
            raise ValueError('Unknown data type for this model.')

        emb1, emb2 = torch.cat((U_src, F_src), dim=1).transpose(1, 2), torch.cat((U_tgt, F_tgt), dim=1).transpose(1, 2)
        ss = []

        if not self.cross_iter:
            # Vanilla PCA-GM
            for i in range(self.gnn_layer):
                gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
                emb1, emb2 = gnn_layer([A_src, emb1], [A_tgt, emb2]) 
                affinity = getattr(self, 'affinity_{}'.format(i))
                s = affinity(emb1, emb2)

                #s = self.sinkhorn(s, ns_src, ns_tgt, dummy_row=True)
                if cfg.normalize_gnn_outputs_channles:
                    s = normalize_over_channels(s)

                if cfg.OPTIMIZATION_METHOD == 'Sinkhorn':
                    if cfg.MATCHING_TYPE == 'Balanced':
                        softs = self.sinkhorn(s, ns_src, ns_tgt, dummy_row=True)
                    elif cfg.MATCHING_TYPE == 'Unbalanced':
                        if cfg.ipf:
                            softs = torch.zeros_like(s)
                            if cfg.train_noise_factor and s.requires_grad:
                                if cfg.samples_per_num_train > 1:
                                    softs = softs.repeat(cfg.samples_per_num_train, 1, 1)
                                    ns_src = ns_src.repeat(cfg.samples_per_num_train, 1)
                                    ns_tgt = ns_tgt.repeat(cfg.samples_per_num_train, 1)
                            for b in range(s.shape[0]):
                                effective_rows = ns_src[b]
                                effective_cols = ns_tgt[b]
                                mat_to_ipf = s[b, :effective_rows, :effective_cols]

                                if cfg.ipf_on_pred:
                                    x_prenorm = hungarian(mat_to_ipf)
                                else:
                                    x_prenorm = gt_perm_mats[0][b, :effective_rows, :effective_cols]

                                row_marginals = torch.sum(x_prenorm, dim=1)
                                epsilon_row_marginals = torch.where(row_marginals > 0, row_marginals, to_var(
                                    torch.tensor(cfg.epsilon_approx, dtype=torch.float)))
                                column_marginals = torch.sum(x_prenorm, dim=0)
                                epsilon_column_marginals = torch.where(column_marginals > 0, column_marginals, to_var(
                                    torch.tensor(cfg.epsilon_approx, dtype=torch.float)))

                                if cfg.train_noise_factor and mat_to_ipf.requires_grad:
                                    sigma_tmp = to_var(torch.ones_like(mat_to_ipf, dtype=torch.float)) / cfg.sigma_norm
                                    n = mat_to_ipf.size()[0]
                                    m = mat_to_ipf.size()[1]
                                    mat_to_ipf = mat_to_ipf.view(-1, n, m)

                                    # reassign mat_to_ipf back to softs
                                    for noise_num in range(cfg.samples_per_num_train):
                                        mat_to_ipf, _ = my_ops.my_phi_and_gamma_sigma_unbalanced(mat_to_ipf, 1,
                                                                                                 cfg.train_noise_factor,
                                                                                                 sigma_tmp)
                                        soft_mat_to_ipf = my_IPF(torch.sigmoid(mat_to_ipf), epsilon_row_marginals,
                                                                 epsilon_column_marginals)
                                        softs[b + noise_num * batch_size, :effective_rows,
                                        :effective_cols] = soft_mat_to_ipf

                                elif mat_to_ipf.requires_grad:
                                    softs[b, :effective_rows, :effective_cols] = my_IPF(torch.sigmoid(mat_to_ipf),
                                                                                        epsilon_row_marginals,
                                                                                        epsilon_column_marginals)
                                    if cfg.calc_MAE:
                                        calc_differences_from_marginals(softs[b,:effective_rows, :effective_cols], effective_rows, row_marginals, effective_cols, column_marginals)

                                elif not mat_to_ipf.requires_grad:  # inference without matching knowledge
                                    if cfg.prediction_type == 'ones_marginals':  # 'sinkhorn', 'norelaxation', 'dustbin', 'ones_marginals'
                                        one_row_marginals = to_var(torch.ones(mat_to_ipf.size()[0]))
                                        one_column_marginals = to_var(torch.ones(mat_to_ipf.size()[1]))
                                        softs[b, :effective_rows, :effective_cols] = my_IPF(torch.sigmoid(mat_to_ipf),
                                                                                            one_row_marginals,
                                                                                            one_column_marginals)

                                    if cfg.prediction_type == 'normonpred':
                                        x_prenorm = hungarian(mat_to_ipf)

                                        row_marginals = torch.sum(x_prenorm, dim=1)
                                        epsilon_row_marginals = torch.where(row_marginals > 0, row_marginals, to_var(
                                            torch.tensor(cfg.epsilon_approx, dtype=torch.float)))
                                        column_marginals = torch.sum(x_prenorm, dim=0)
                                        epsilon_column_marginals = torch.where(column_marginals > 0, column_marginals,
                                                                               to_var(torch.tensor(cfg.epsilon_approx,
                                                                                                   dtype=torch.float)))

                                        softs[b, :effective_rows, :effective_cols] = my_IPF(torch.sigmoid(mat_to_ipf),
                                                                                            epsilon_row_marginals,
                                                                                            epsilon_column_marginals)

                                    # without relaxation
                                    if cfg.prediction_type == 'norelaxation':
                                        softs = s

                                    if cfg.prediction_type == 'sinkhorn':
                                        softs = self.sinkhorn(s, ns_src, ns_tgt, dummy_row=True)

                                    if cfg.prediction_type == 'row_normalization':
                                        softs = torch.zeros_like(s)
                                        for b in range(s.shape[0]):
                                            effective_rows = ns_src[b]
                                            effective_cols = ns_tgt[b]
                                            mat = torch.sigmoid(
                                                s[b, :effective_rows, :effective_cols].view(-1, effective_rows,
                                                                                            effective_cols))
                                            softs[b, :effective_rows, :effective_cols] = mat / mat.sum(dim=2,
                                                                                                       keepdim=True)

                                    if cfg.prediction_type == 'dustbin':
                                        softs = torch.zeros_like(s)
                                        for b in range(s.shape[0]):
                                            effective_rows = ns_src[b]
                                            effective_cols = ns_tgt[b]
                                            # Run the optimal transport.
                                            mat = s[b, :effective_rows, :effective_cols].view(-1, effective_rows,
                                                                                              effective_cols)
                                            scores = log_optimal_transport(mat, self.bin_score, iters=25)
                                            ss = torch.sigmoid(scores)
                                            softs[b, :effective_rows, :effective_cols] = ss[:, :-1, :-1]

                                    if cfg.prediction_type == 'gt_marginals':
                                        for b in range(s.shape[0]):
                                            effective_rows = ns_src[b]
                                            effective_cols = ns_tgt[b]
                                            mat_to_ipf = s[b, :effective_rows, :effective_cols]
                                            gt_premarginal_assignments = gt_perm_mats[0][b, :effective_rows,
                                                                         :effective_cols]
                                            row_marginals = torch.sum(gt_premarginal_assignments, dim=1)
                                            epsilon_row_marginals = torch.where(row_marginals > 0, row_marginals,
                                                                                to_var(torch.tensor(1e-7,
                                                                                                    dtype=torch.float)))
                                            column_marginals = torch.sum(gt_premarginal_assignments, dim=0)
                                            epsilon_column_marginals = torch.where(column_marginals > 0,
                                                                                   column_marginals, to_var(
                                                    torch.tensor(1e-7, dtype=torch.float)))

                                            softs[b, :effective_rows, :effective_cols] = my_IPF(
                                                torch.sigmoid(mat_to_ipf), epsilon_row_marginals,
                                                epsilon_column_marginals)

                        else:
                            softs = self.sinkhorn(s, ns_src, ns_tgt, dummy_row=True)
                            for b in range(s.shape[0]):
                                effective_rows = ns_src[b]
                                effective_cols = ns_tgt[b]
                                x_prenorm = hungarian(s[b, :effective_rows, :effective_cols])
                                row_marginals = torch.sum(x_prenorm, dim=1)
                                column_marginals = torch.sum(x_prenorm, dim=0)
                                if cfg.calc_MAE:
                                    calc_differences_from_marginals(
                                        softs[b, :effective_rows, :effective_cols], effective_rows, row_marginals,
                                        effective_cols, column_marginals)

                elif cfg.OPTIMIZATION_METHOD == 'superglue':
                    softs = torch.zeros_like(s)
                    if cfg.train_noise_factor and s.requires_grad:
                        if cfg.samples_per_num_train > 1:
                            softs = softs.repeat(cfg.samples_per_num_train, 1, 1)
                    '''
                    # Run the optimal transport.
                    scores = log_optimal_transport(
                        s, self.bin_score,
                        iters=100)
                    ss = torch.exp(scores)
                    softs = ss[:, :-1, :-1]
                    '''
                    for b in range(s.shape[0]):
                        effective_rows = ns_src[b]
                        effective_cols = ns_tgt[b]
                        x_prenorm = hungarian(s[b, :effective_rows, :effective_cols])
                        row_marginals = torch.sum(x_prenorm, dim=1)
                        column_marginals = torch.sum(x_prenorm, dim=0)

                        # Run the optimal transport.
                        mat = s[b, :effective_rows, :effective_cols].view(-1, effective_rows, effective_cols)
                        scores = log_optimal_transport(mat, self.bin_score, iters=25)
                        ss_glue = torch.exp(scores)
                        softs[b, :effective_rows, :effective_cols] = ss_glue[:, :-1, :-1]
                        if cfg.calc_MAE:
                            calc_differences_from_marginals(softs[b, :effective_rows, :effective_cols],
                                                                   effective_rows, row_marginals.unsqueeze(0), effective_cols, column_marginals.unsqueeze(0))



                if cfg.calc_pred_shift:
                    for b in range(batch_size):
                        calc_prediction_shift(s[b, :ns_src[b], :ns_tgt[b]], softs[b, :ns_src[b], :ns_tgt[b]])

                #ss.append(s)
                ss.append(softs)

                if i == self.gnn_layer - 2:
                    cross_graph = getattr(self, 'cross_graph_{}'.format(i))
                    #new_emb1 = cross_graph(torch.cat((emb1, torch.bmm(s, emb2)), dim=-1))
                    #new_emb2 = cross_graph(torch.cat((emb2, torch.bmm(s.transpose(1, 2), emb1)), dim=-1))
                    new_emb1 = cross_graph(torch.cat((emb1, torch.bmm(softs, emb2)), dim=-1))
                    new_emb2 = cross_graph(torch.cat((emb2, torch.bmm(softs.transpose(1, 2), emb1)), dim=-1))
                    emb1 = new_emb1
                    emb2 = new_emb2
        else:
            # IPCA-GM
            for i in range(self.gnn_layer - 1):
                gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
                emb1, emb2 = gnn_layer([A_src, emb1], [A_tgt, emb2])

            emb1_0, emb2_0 = emb1, emb2
            s = torch.zeros(emb1.shape[0], emb1.shape[1], emb2.shape[1], device=emb1.device)

            for x in range(self.cross_iter_num):
                i = self.gnn_layer - 2
                cross_graph = getattr(self, 'cross_graph_{}'.format(i))
                emb1 = cross_graph(torch.cat((emb1_0, torch.bmm(s, emb2_0)), dim=-1))
                emb2 = cross_graph(torch.cat((emb2_0, torch.bmm(s.transpose(1, 2), emb1_0)), dim=-1))

                i = self.gnn_layer - 1
                gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
                emb1, emb2 = gnn_layer([A_src, emb1], [A_tgt, emb2])
                affinity = getattr(self, 'affinity_{}'.format(i))
                s = affinity(emb1, emb2)
                if cfg.normalize_gnn_outputs_channles:
                    s = normalize_over_channels(s)
                '''
                if cfg.OPTIMIZATION_METHOD =='Sinkhorn':
                    s = self.sinkhorn(s, ns_src, ns_tgt, dummy_row=True)
                '''
                if cfg.OPTIMIZATION_METHOD == 'Sinkhorn':
                    if cfg.MATCHING_TYPE == 'Balanced':
                        softs = self.sinkhorn(s, ns_src, ns_tgt, dummy_row=True)
                    elif cfg.MATCHING_TYPE == 'Unbalanced':
                        if cfg.ipf:
                            softs = torch.zeros_like(s)
                            if cfg.train_noise_factor and s.requires_grad:
                                if cfg.samples_per_num_train > 1:
                                    softs = softs.repeat(cfg.samples_per_num_train, 1, 1)
                            for b in range(s.shape[0]):
                                effective_rows = ns_src[b]
                                effective_cols = ns_tgt[b]
                                mat_to_ipf = s[b, :effective_rows, :effective_cols]

                                if cfg.ipf_on_pred:
                                    x_prenorm = hungarian(mat_to_ipf)
                                else:
                                    x_prenorm = gt_perm_mats[0][b, :effective_rows, :effective_cols]

                                row_marginals = torch.sum(x_prenorm, dim=1)
                                epsilon_row_marginals = torch.where(row_marginals > 0, row_marginals, to_var(
                                    torch.tensor(cfg.epsilon_approx, dtype=torch.float)))
                                column_marginals = torch.sum(x_prenorm, dim=0)
                                epsilon_column_marginals = torch.where(column_marginals > 0, column_marginals, to_var(
                                    torch.tensor(cfg.epsilon_approx, dtype=torch.float)))

                                if cfg.train_noise_factor and mat_to_ipf.requires_grad:
                                    sigma_tmp = to_var(torch.ones_like(mat_to_ipf, dtype=torch.float)) / cfg.sigma_norm
                                    n = mat_to_ipf.size()[0]
                                    m = mat_to_ipf.size()[1]
                                    mat_to_ipf = mat_to_ipf.view(-1, n, m)

                                    # reassign mat_to_ipf back to softs
                                    for noise_num in range(cfg.samples_per_num_train):
                                        mat_to_ipf, _ = my_ops.my_phi_and_gamma_sigma_unbalanced(mat_to_ipf, 1,
                                                                                                 cfg.train_noise_factor,
                                                                                                 sigma_tmp)
                                        soft_mat_to_ipf = my_IPF(torch.exp(mat_to_ipf), epsilon_row_marginals,
                                                                 epsilon_column_marginals)
                                        softs[b + noise_num * batch_size, :effective_rows,
                                        :effective_cols] = soft_mat_to_ipf

                                elif mat_to_ipf.requires_grad:
                                    softs[b, :effective_rows, :effective_cols] = my_IPF(torch.sigmoid(mat_to_ipf),
                                                                                        epsilon_row_marginals,
                                                                                        epsilon_column_marginals)
                                    if cfg.calc_MAE:
                                        calc_differences_from_marginals(softs[b,:effective_rows, :effective_cols], effective_rows, row_marginals, effective_cols, column_marginals)

                                elif not mat_to_ipf.requires_grad:  # inference without matching knowledge
                                    if cfg.prediction_type == 'ones_marginals':  # 'sinkhorn', 'norelaxation', 'dustbin', 'ones_marginals'
                                        one_row_marginals = to_var(torch.ones(mat_to_ipf.size()[0]))
                                        one_column_marginals = to_var(torch.ones(mat_to_ipf.size()[1]))
                                        softs[b, :effective_rows, :effective_cols] = my_IPF(torch.sigmoid(mat_to_ipf),
                                                                                            one_row_marginals,
                                                                                            one_column_marginals)

                                    if cfg.prediction_type == 'normonpred':
                                        x_prenorm = hungarian(mat_to_ipf)

                                        row_marginals = torch.sum(x_prenorm, dim=1)
                                        epsilon_row_marginals = torch.where(row_marginals > 0, row_marginals, to_var(
                                            torch.tensor(cfg.epsilon_approx, dtype=torch.float)))
                                        column_marginals = torch.sum(x_prenorm, dim=0)
                                        epsilon_column_marginals = torch.where(column_marginals > 0, column_marginals,
                                                                               to_var(torch.tensor(cfg.epsilon_approx,
                                                                                                   dtype=torch.float)))

                                        softs[b, :effective_rows, :effective_cols] = my_IPF(torch.sigmoid(mat_to_ipf),
                                                                                            epsilon_row_marginals,
                                                                                            epsilon_column_marginals)

                                    # without relaxation
                                    if cfg.prediction_type == 'norelaxation':
                                        softs = s

                                    if cfg.prediction_type == 'sinkhorn':
                                        softs = self.sinkhorn(s, ns_src, ns_tgt, dummy_row=True)

                                    if cfg.prediction_type == 'row_normalization':
                                        softs = torch.zeros_like(s)
                                        for b in range(s.shape[0]):
                                            effective_rows = ns_src[b]
                                            effective_cols = ns_tgt[b]
                                            mat = torch.sigmoid(
                                                s[b, :effective_rows, :effective_cols].view(-1, effective_rows,
                                                                                            effective_cols))
                                            softs[b, :effective_rows, :effective_cols] = mat / mat.sum(dim=2,
                                                                                                       keepdim=True)

                                    if cfg.prediction_type == 'dustbin':
                                        softs = torch.zeros_like(s)
                                        for b in range(s.shape[0]):
                                            effective_rows = ns_src[b]
                                            effective_cols = ns_tgt[b]
                                            # Run the optimal transport.
                                            mat = s[b, :effective_rows, :effective_cols].view(-1, effective_rows,
                                                                                              effective_cols)
                                            scores = log_optimal_transport(mat, self.bin_score, iters=25)
                                            ss = torch.sigmoid(scores)
                                            softs[b, :effective_rows, :effective_cols] = ss[:, :-1, :-1]

                                    if cfg.prediction_type == 'gt_marginals':
                                        for b in range(s.shape[0]):
                                            effective_rows = ns_src[b]
                                            effective_cols = ns_tgt[b]
                                            mat_to_ipf = s[b, :effective_rows, :effective_cols]
                                            gt_premarginal_assignments = gt_perm_mats[0][b, :effective_rows,
                                                                         :effective_cols]
                                            row_marginals = torch.sum(gt_premarginal_assignments, dim=1)
                                            epsilon_row_marginals = torch.where(row_marginals > 0, row_marginals,
                                                                                to_var(torch.tensor(1e-7,
                                                                                                    dtype=torch.float)))
                                            column_marginals = torch.sum(gt_premarginal_assignments, dim=0)
                                            epsilon_column_marginals = torch.where(column_marginals > 0,
                                                                                   column_marginals, to_var(
                                                    torch.tensor(1e-7, dtype=torch.float)))

                                            softs[b, :effective_rows, :effective_cols] = my_IPF(
                                                torch.sigmoid(mat_to_ipf), epsilon_row_marginals,
                                                epsilon_column_marginals)

                        else:
                            softs = self.sinkhorn(s, ns_src, ns_tgt, dummy_row=True)
                            for b in range(s.shape[0]):
                                effective_rows = ns_src[b]
                                effective_cols = ns_tgt[b]
                                x_prenorm = hungarian(s[b, :effective_rows, :effective_cols])
                                row_marginals = torch.sum(x_prenorm, dim=1)
                                column_marginals = torch.sum(x_prenorm, dim=0)
                                if cfg.calc_MAE:
                                    calc_differences_from_marginals(
                                        softs[b, :effective_rows, :effective_cols], effective_rows, row_marginals,
                                        effective_cols, column_marginals)

                elif cfg.OPTIMIZATION_METHOD == 'superglue':
                    softs = torch.zeros_like(s)
                    if cfg.train_noise_factor and s.requires_grad:
                        if cfg.samples_per_num_train > 1:
                            softs = softs.repeat(cfg.samples_per_num_train, 1, 1)
                    '''
                    # Run the optimal transport.
                    scores = log_optimal_transport(
                        s, self.bin_score,
                        iters=100)
                    ss = torch.exp(scores)
                    softs = ss[:, :-1, :-1]
                    '''
                    for b in range(s.shape[0]):
                        effective_rows = ns_src[b]
                        effective_cols = ns_tgt[b]
                        x_prenorm = hungarian(s[b, :effective_rows, :effective_cols])
                        row_marginals = torch.sum(x_prenorm, dim=1)
                        column_marginals = torch.sum(x_prenorm, dim=0)

                        # Run the optimal transport.
                        mat = s[b, :effective_rows, :effective_cols].view(-1, effective_rows, effective_cols)
                        scores = log_optimal_transport(mat, self.bin_score, iters=25)
                        ss_scores = torch.exp(scores)
                        softs[b, :effective_rows, :effective_cols] = ss_scores[:, :-1, :-1]
                        if cfg.calc_MAE:
                            calc_differences_from_marginals(softs[b, :effective_rows, :effective_cols],
                                                                   effective_rows, row_marginals, effective_cols, column_marginals)
                if cfg.train_noise_factor:
                    if cfg.samples_per_num_train > 1:
                        ns_src = ns_src.repeat(cfg.samples_per_num_train)
                        ns_tgt = ns_tgt.repeat(cfg.samples_per_num_train)
                if cfg.calc_pred_shift:
                    for b in range(batch_size):
                        calc_prediction_shift(s[b, :ns_src[b], :ns_tgt[b]], softs[b, :ns_src[b], :ns_tgt[b]])

                #ss.append(s)
                ss.append(softs)

        data_dict.update({
            'ds_mat': ss[-1],
            'perm_mat': hungarian(ss[-1], ns_src, ns_tgt)
        })
        return data_dict
