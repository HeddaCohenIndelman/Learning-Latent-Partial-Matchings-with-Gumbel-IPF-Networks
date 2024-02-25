import torch
import itertools

from models.BBGM.affinity_layer import InnerProductWithWeightsAffinity
from models.BBGM.sconv_archs import SiameseSConvOnNodes, SiameseNodeFeaturesToEdgeFeatures
from src.feature_align import feature_align
from src.factorize_graph_matching import construct_aff_mat
from src.utils.pad_tensor import pad_tensor
from models.NGM.gnn import GNNLayer
from src.lap_solvers.sinkhorn import Sinkhorn
from src.lap_solvers.hungarian import hungarian
import torch_geometric
import numpy

import my_ops
from my_ops import IPF, my_IPF


from src.utils.config import cfg

from src.backbone import *


is_cuda = torch.cuda.is_available()
def to_var(x):
    if is_cuda:
        x = x.cuda()
    return x
CNN = eval(cfg.BACKBONE)

def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1


def lexico_iter(lex):
    return itertools.combinations(lex, 2)


def normalize_over_channels(x):
    channel_norms = torch.norm(x, dim=1, keepdim=True)
    return x / channel_norms


def concat_features(embeddings, num_vertices):
    res = torch.cat([embedding[:, :num_v] for embedding, num_v in zip(embeddings, num_vertices)], dim=-1)
    return res.transpose(0, 1)

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
    with open('restats_MAE_ptype'+cfg.OPTIMIZATION_METHOD+'_ipf_type'+str(cfg.ipf)+'_src_outlier'+str(cfg.PROBLEM.SRC_OUTLIER)+'_trg_outlier'+str(cfg.PROBLEM.TGT_OUTLIER)+ 'partial_matching_'+str(partial_matching)+'_average_diff_abs_marginals','a') as file_diff:
        file_diff.write("{:.6f}".format(rows_avg_absdiff) +'\t' + str(n_rows.item())+'\t' +"{:.6f}".format(cols_avg_absdiff) +'\t' + str(n_cols.item())+'\n')
    return avg_diff

def calc_prediction_shift(unnormalized_s, normalized_s):

    pred_unnormalized_s = hungarian(unnormalized_s)
    pred_normalized_s = hungarian(normalized_s)

    m = unnormalized_s.size(dim=0)
    n = unnormalized_s.size(dim=1)

    cardinality = torch.tensor(min(m, n))

    prediction_shift = cardinality - torch.sum(torch.logical_and(pred_unnormalized_s == 1, pred_normalized_s == 1))
    prediction_shift_file = 'restats_prediction_shift_ptype'+cfg.OPTIMIZATION_METHOD+'_ipf_'+str(cfg.ipf)+'.txt'
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

class Net(CNN):
    def __init__(self):
        super(Net, self).__init__()
        self.message_pass_node_features = SiameseSConvOnNodes(input_node_dim=cfg.NGM.FEATURE_CHANNEL * 2)
        self.build_edge_features_from_node_features = SiameseNodeFeaturesToEdgeFeatures(
            total_num_nodes=self.message_pass_node_features.num_node_features
        )
        self.global_state_dim = cfg.NGM.FEATURE_CHANNEL * 2
        self.vertex_affinity = InnerProductWithWeightsAffinity(
            self.global_state_dim, self.message_pass_node_features.num_node_features)
        self.edge_affinity = InnerProductWithWeightsAffinity(
            self.global_state_dim,
            self.build_edge_features_from_node_features.num_edge_features)

        self.rescale = cfg.PROBLEM.RESCALE
        self.tau = cfg.NGM.SK_TAU
        self.mgm_tau = cfg.NGM.MGM_SK_TAU
        self.univ_size = cfg.NGM.UNIV_SIZE

        self.sinkhorn = Sinkhorn(max_iter=cfg.NGM.SK_ITER_NUM, tau=self.tau, epsilon=cfg.NGM.SK_EPSILON)
        self.sinkhorn_mgm = Sinkhorn(max_iter=cfg.NGM.SK_ITER_NUM, epsilon=cfg.NGM.SK_EPSILON, tau=self.mgm_tau)
        self.gnn_layer = cfg.NGM.GNN_LAYER
        for i in range(self.gnn_layer):
            tau = cfg.NGM.SK_TAU
            if i == 0:
                gnn_layer = GNNLayer(1, 1,
                                     cfg.NGM.GNN_FEAT[i] + cfg.NGM.SK_EMB, cfg.NGM.GNN_FEAT[i],
                                     sk_channel=cfg.NGM.SK_EMB, sk_tau=tau, edge_emb=cfg.NGM.EDGE_EMB)
            else:
                gnn_layer = GNNLayer(cfg.NGM.GNN_FEAT[i - 1] + cfg.NGM.SK_EMB, cfg.NGM.GNN_FEAT[i - 1],
                                     cfg.NGM.GNN_FEAT[i] + cfg.NGM.SK_EMB, cfg.NGM.GNN_FEAT[i],
                                     sk_channel=cfg.NGM.SK_EMB, sk_tau=tau, edge_emb=cfg.NGM.EDGE_EMB)

            self.add_module('gnn_layer_{}'.format(i), gnn_layer)

        self.classifier = nn.Linear(cfg.NGM.GNN_FEAT[-1] + cfg.NGM.SK_EMB, 1)

        bin_score = torch.nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)


    def forward(
        self,
        data_dict,
    ):
        images = data_dict['images']
        points = data_dict['Ps']
        n_points = data_dict['ns']
        graphs = data_dict['pyg_graphs']
        batch_size = data_dict['batch_size']
        num_graphs = len(images)

        if cfg.PROBLEM.TYPE == '2GM' and 'gt_perm_mat' in data_dict:
            gt_perm_mats = [data_dict['gt_perm_mat']]
        elif cfg.PROBLEM.TYPE == 'MGM' and 'gt_perm_mat' in data_dict:

            if cfg.MATCHING_TYPE == "Balanced":
                perm_mat_list = data_dict['gt_perm_mat']
                gt_perm_mats = [torch.bmm(pm_src, pm_tgt.transpose(1, 2)) for pm_src, pm_tgt in lexico_iter(perm_mat_list)]
            elif cfg.MATCHING_TYPE == "Unbalanced":
                gt_perm_mats = data_dict['gt_perm_mat']


        else:
            raise ValueError('Ground truth information is required during training.')

        global_list = []
        orig_graph_list = []
        for image, p, n_p, graph in zip(images, points, n_points, graphs):
            # extract feature
            nodes = self.node_layers(image)
            edges = self.edge_layers(nodes)

            global_list.append(self.final_layers(edges).reshape((nodes.shape[0], -1)))

            nodes = normalize_over_channels(nodes)
            edges = normalize_over_channels(edges)

            # arrange features
            U = concat_features(feature_align(nodes, p, n_p, self.rescale), n_p)
            F = concat_features(feature_align(edges, p, n_p, self.rescale), n_p)
            node_features = torch.cat((U, F), dim=1)
            graph.x = node_features

            graph = self.message_pass_node_features(graph)
            orig_graph = self.build_edge_features_from_node_features(graph)
            orig_graph_list.append(orig_graph)

        global_weights_list = [
            torch.cat([global_src, global_tgt], axis=-1) for global_src, global_tgt in lexico_iter(global_list)
        ]

        global_weights_list = [normalize_over_channels(g) for g in global_weights_list]

        #same as (minus) unary_costs_list in Rolinek
        unary_affs_list = [
            self.vertex_affinity([item.x for item in g_1], [item.x for item in g_2], global_weights)
            for (g_1, g_2), global_weights in zip(lexico_iter(orig_graph_list), global_weights_list)
        ]

        #same as (minus) quadratic_costs_list  in Rolinek
        quadratic_affs_list = [
            self.edge_affinity([item.edge_attr for item in g_1], [item.edge_attr for item in g_2], global_weights)
            for (g_1, g_2), global_weights in zip(lexico_iter(orig_graph_list), global_weights_list)
        ]

        quadratic_affs_list = [[0.5 * x for x in quadratic_affs] for quadratic_affs in quadratic_affs_list]


        s_list, mgm_s_list, x_list, mgm_x_list, indices = [], [], [], [], []
        row_x_list = []

        for unary_affs, quadratic_affs, (idx1, idx2) in zip(unary_affs_list, quadratic_affs_list, lexico_iter(range(num_graphs))):
            kro_G, kro_H = data_dict['KGHs'] if num_graphs == 2 else data_dict['KGHs']['{},{}'.format(idx1, idx2)]
            Kp = torch.stack(pad_tensor(unary_affs), dim=0)
            Ke = torch.stack(pad_tensor(quadratic_affs), dim=0)
            K = construct_aff_mat(Ke, Kp, kro_G, kro_H)


            if num_graphs == 2: data_dict['aff_mat'] = K

            if cfg.NGM.FIRST_ORDER:
                emb = Kp.transpose(1, 2).contiguous().view(Kp.shape[0], -1, 1)
            else:
                emb = torch.ones(K.shape[0], K.shape[1], 1, device=K.device)

            if cfg.NGM.POSITIVE_EDGES:
                A = (K > 0).to(K.dtype)
            else:
                A = (K != 0).to(K.dtype)

            emb_K = K.unsqueeze(-1)

            # NGM qap solver
            for i in range(self.gnn_layer):
                gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
                emb_K, emb = gnn_layer(A, emb_K, emb, n_points[idx1], n_points[idx2])

            v = self.classifier(emb)

            if cfg.normalize_gnn_outputs_channles:
                v = normalize_over_channels(v)

            s = v.view(v.shape[0], points[idx2].shape[1], -1).transpose(1, 2)

            #if cfg.OPTIMIZATION_METHOD == 'Sinkhorn':
            #    ss = self.sinkhorn(s, n_points[idx1], n_points[idx2], dummy_row=True)

            if cfg.OPTIMIZATION_METHOD == 'Sinkhorn':
                if cfg.MATCHING_TYPE == 'Balanced':
                    softs = self.sinkhorn(s, n_points[idx1], n_points[idx2], dummy_row=True)
                elif cfg.MATCHING_TYPE == 'Unbalanced':
                    if cfg.ipf:
                        softs = torch.zeros_like(s)
                        if cfg.train_noise_factor and s.requires_grad:
                            if cfg.samples_per_num_train > 1:
                                softs = softs.repeat(cfg.samples_per_num_train, 1, 1)
                        for b in range(s.shape[0]):
                            effective_rows = n_points[idx1][b]
                            effective_cols = n_points[idx2][b]
                            mat_to_ipf = s[b,:effective_rows, :effective_cols]

                            if cfg.ipf_on_pred:
                                x_prenorm = hungarian(mat_to_ipf)
                            else:
                                x_prenorm = gt_perm_mats[0][b,:effective_rows,:effective_cols]

                            row_marginals = torch.sum(x_prenorm, dim=1)
                            epsilon_row_marginals = torch.where(row_marginals > 0, row_marginals, to_var(torch.tensor(cfg.epsilon_approx, dtype=torch.float)))
                            column_marginals = torch.sum(x_prenorm, dim=0)
                            epsilon_column_marginals = torch.where(column_marginals > 0, column_marginals, to_var(torch.tensor(cfg.epsilon_approx, dtype=torch.float)))

                            if cfg.train_noise_factor and mat_to_ipf.requires_grad:
                                sigma_tmp = to_var(torch.ones_like(mat_to_ipf, dtype=torch.float)) / cfg.sigma_norm
                                n = mat_to_ipf.size()[0]
                                m = mat_to_ipf.size()[1]
                                mat_to_ipf = mat_to_ipf.view(-1, n, m)

                                #reassign mat_to_ipf back to softs
                                for noise_num in range(cfg.samples_per_num_train):
                                    mat_to_ipf, _ = my_ops.my_phi_and_gamma_sigma_unbalanced(mat_to_ipf, 1, cfg.train_noise_factor, sigma_tmp)
                                    soft_mat_to_ipf = my_IPF(torch.sigmoid(mat_to_ipf), epsilon_row_marginals, epsilon_column_marginals)
                                    softs[b+noise_num*batch_size,:effective_rows, :effective_cols] = soft_mat_to_ipf

                            elif mat_to_ipf.requires_grad:
                                softs[b,:effective_rows, :effective_cols] = my_IPF(torch.sigmoid(mat_to_ipf), epsilon_row_marginals, epsilon_column_marginals)
                                if cfg.calc_MAE:
                                    avg_diff = calc_differences_from_marginals(softs[b,:effective_rows, :effective_cols], effective_rows, row_marginals, effective_cols, column_marginals)

                            elif not mat_to_ipf.requires_grad: # inference without matching knowledge
                                if cfg.prediction_type == 'ones_marginals':  # 'sinkhorn', 'norelaxation', 'dustbin', 'ones_marginals'
                                    one_row_marginals = to_var(torch.ones(mat_to_ipf.size()[0]))
                                    one_column_marginals = to_var(torch.ones(mat_to_ipf.size()[1]))
                                    softs[b, :effective_rows, :effective_cols] = my_IPF(torch.sigmoid(mat_to_ipf), one_row_marginals, one_column_marginals)

                                if cfg.prediction_type == 'normonpred':
                                    x_prenorm = hungarian(mat_to_ipf)

                                    row_marginals = torch.sum(x_prenorm, dim=1)
                                    epsilon_row_marginals = torch.where(row_marginals > 0, row_marginals, to_var(torch.tensor(cfg.epsilon_approx, dtype=torch.float)))
                                    column_marginals = torch.sum(x_prenorm, dim=0)
                                    epsilon_column_marginals = torch.where(column_marginals > 0, column_marginals, to_var(torch.tensor(cfg.epsilon_approx, dtype=torch.float)))

                                    softs[b, :effective_rows, :effective_cols] = my_IPF(torch.sigmoid(mat_to_ipf), epsilon_row_marginals,epsilon_column_marginals)

                                # without relaxation
                                if cfg.prediction_type == 'norelaxation':
                                    softs = s

                                if cfg.prediction_type == 'sinkhorn':
                                    softs = self.sinkhorn(s, n_points[idx1], n_points[idx2], dummy_row=True)

                                if cfg.prediction_type == 'row_normalization':
                                    softs = torch.zeros_like(s)
                                    for b in range(s.shape[0]):
                                        effective_rows = n_points[idx1][b]
                                        effective_cols = n_points[idx2][b]
                                        mat = torch.sigmoid(s[b, :effective_rows, :effective_cols].view(-1, effective_rows, effective_cols))
                                        softs[b, :effective_rows, :effective_cols] = mat/mat.sum(dim=2, keepdim=True)

                                if cfg.prediction_type == 'dustbin':
                                    softs = torch.zeros_like(s)
                                    for b in range(s.shape[0]):
                                        effective_rows = n_points[idx1][b]
                                        effective_cols = n_points[idx2][b]
                                        # Run the optimal transport.
                                        mat = s[b, :effective_rows, :effective_cols].view(-1, effective_rows, effective_cols)
                                        scores = log_optimal_transport(mat, self.bin_score, iters=25)
                                        ss = torch.sigmoid(scores)
                                        softs[b, :effective_rows, :effective_cols] = ss[:, :-1, :-1]

                                if cfg.prediction_type == 'gt_marginals':
                                    for b in range(s.shape[0]):
                                        effective_rows = n_points[idx1][b]
                                        effective_cols = n_points[idx2][b]
                                        mat_to_ipf = s[b, :effective_rows, :effective_cols]
                                        gt_premarginal_assignments = gt_perm_mats[0][b, :effective_rows, :effective_cols]
                                        row_marginals = torch.sum(gt_premarginal_assignments, dim=1)
                                        epsilon_row_marginals = torch.where(row_marginals > 0, row_marginals, to_var(torch.tensor(1e-7, dtype=torch.float)))
                                        column_marginals = torch.sum(gt_premarginal_assignments, dim=0)
                                        epsilon_column_marginals = torch.where(column_marginals > 0, column_marginals, to_var(torch.tensor(1e-7, dtype=torch.float)))

                                        softs[b, :effective_rows, :effective_cols] = my_IPF(torch.sigmoid(mat_to_ipf), epsilon_row_marginals,  epsilon_column_marginals)

                    else:
                        softs = self.sinkhorn(s, n_points[idx1], n_points[idx2], dummy_row=True)
                        for b in range(s.shape[0]):
                            effective_rows = n_points[idx1][b]
                            effective_cols = n_points[idx2][b]
                            x_prenorm = hungarian(s[b,:effective_rows, :effective_cols])
                            row_marginals = torch.sum(x_prenorm, dim=1)
                            column_marginals = torch.sum(x_prenorm, dim=0)
                            if cfg.calc_MAE:
                                avg_diff = calc_differences_from_marginals(softs[b, :effective_rows, :effective_cols], effective_rows, row_marginals, effective_cols, column_marginals)

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
                    effective_rows = n_points[idx1][b]
                    effective_cols = n_points[idx2][b]
                    x_prenorm = hungarian(s[b,:effective_rows, :effective_cols])
                    row_marginals = torch.sum(x_prenorm, dim=1)
                    column_marginals = torch.sum(x_prenorm, dim=0)

                    # Run the optimal transport.
                    mat = s[b, :effective_rows, :effective_cols].view(-1, effective_rows, effective_cols)
                    scores = log_optimal_transport(mat, self.bin_score, iters=25)
                    ss = torch.exp(scores)
                    softs[b,:effective_rows, :effective_cols] = ss[:, :-1, :-1]
                    if cfg.calc_MAE:
                        avg_diff = calc_differences_from_marginals(softs[b, :effective_rows, :effective_cols], effective_rows, row_marginals, effective_cols, column_marginals)

            elif cfg.OPTIMIZATION_METHOD == 'Direct':

                if cfg.MATCHING_TYPE == 'Balanced':
                    ss = self.sinkhorn(s, n_points[idx1], n_points[idx2], dummy_row=True)
                elif cfg.MATCHING_TYPE == 'Unbalanced':
                    ss = s
            if cfg.train_noise_factor:
                if cfg.samples_per_num_train > 1:
                    n_points[idx1] = n_points[idx1].repeat(cfg.samples_per_num_train)
                    n_points[idx2]= n_points[idx2].repeat(cfg.samples_per_num_train)
            if cfg.calc_pred_shift:
                for b in range(batch_size):
                    calc_prediction_shift(s[b,:n_points[idx1][b], :n_points[idx2][b]], softs[b,:n_points[idx1][b], :n_points[idx2][b]])

            x = hungarian(softs, n_points[idx1], n_points[idx2])

            s_list.append(softs)
            x_list.append(x)

            indices.append((idx1, idx2))

        if num_graphs > 2:
            if cfg.OPTIMIZATION_METHOD == 'Sinkhorn':
                joint_indices = torch.cat((torch.cumsum(torch.stack([torch.max(np) for np in n_points]), dim=0), torch.zeros((1,), dtype=torch.long, device=K.device)))
                joint_S = torch.zeros(batch_size, torch.max(joint_indices), torch.max(joint_indices), device=K.device)
                for idx in range(num_graphs):
                    for b in range(batch_size):
                        start = joint_indices[idx-1]
                        joint_S[b, start:start+n_points[idx][b], start:start+n_points[idx][b]] += torch.eye(n_points[idx][b], device=K.device)

                for (idx1, idx2), s in zip(indices, s_list):
                    if idx1 > idx2:
                        joint_S[:, joint_indices[idx2-1]:joint_indices[idx2], joint_indices[idx1-1]:joint_indices[idx1]] += s.transpose(1, 2)
                    else:
                        joint_S[:, joint_indices[idx1-1]:joint_indices[idx1], joint_indices[idx2-1]:joint_indices[idx2]] += s

                matching_s = []
                for b in range(batch_size):
                    upper = True
                    #e, v = torch.symeig(joint_S[b], eigenvectors=True) #torch.symeig() is deprecated in favor of torch.linalg.eigh() and will be removed in a future PyTorch release.
                    #https://pytorch.org/docs/stable/generated/torch.symeig.html
                    e, v = torch.linalg.eigh(joint_S[b], UPLO='U' if upper else 'L')
                    diff = e[-self.univ_size:-1] - e[-self.univ_size+1:]
                    if self.training and torch.min(torch.abs(diff)) <= 1e-4:
                        matching_s.append(joint_S[b])
                    else:
                        matching_s.append(num_graphs * torch.mm(v[:, -self.univ_size:], v[:, -self.univ_size:].transpose(0, 1)))

                matching_s = torch.stack(matching_s, dim=0)

                for idx1, idx2 in indices:
                    s = matching_s[:, joint_indices[idx1-1]:joint_indices[idx1], joint_indices[idx2-1]:joint_indices[idx2]]
                    if cfg.OPTIMIZATION_METHOD == 'Sinkhorn':
                        s = self.sinkhorn_mgm(torch.log(torch.relu(s)), n_points[idx1], n_points[idx2]) # only perform row/col norm, do not perform exp

                    elif cfg.OPTIMIZATION_METHOD == 'Direct':
                        s = self.sinkhorn_mgm(torch.log(torch.relu(s)), n_points[idx1], n_points[idx2]) # only perform row/col norm, do not perform exp

                    x = hungarian(s, n_points[idx1], n_points[idx2])

                    mgm_x_list.append(x)
                    mgm_s_list.append(s)
            elif cfg.OPTIMIZATION_METHOD == 'Direct':
                mgm_x_list = x_list
                mgm_s_list = s_list
                

        if cfg.PROBLEM.TYPE == '2GM':
            data_dict.update({
                'ds_mat': s_list[0],
                'perm_mat': x_list[0]
            })
            
        elif cfg.PROBLEM.TYPE == 'MGM':
            data_dict.update({
                'ds_mat_list': mgm_s_list,
                'perm_mat_list': mgm_x_list,
                'graph_indices': indices,
                'gt_perm_mat_list': gt_perm_mats
            })

        return data_dict
