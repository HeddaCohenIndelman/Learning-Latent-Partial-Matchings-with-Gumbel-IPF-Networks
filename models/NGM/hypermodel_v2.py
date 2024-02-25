import torch
import itertools

from models.BBGM.affinity_layer import InnerProductWithWeightsAffinity
from models.BBGM.sconv_archs import SiameseSConvOnNodes, SiameseNodeFeaturesToEdgeFeatures
from src.feature_align import feature_align
from src.factorize_graph_matching import construct_aff_mat
from models.NGM.gnn import HyperGNNLayer
from src.lap_solvers.sinkhorn import Sinkhorn
from src.lap_solvers.hungarian import hungarian
from src.utils.pad_tensor import pad_tensor
from src.utils.sparse import to_sparse

from src.utils.config import cfg

from src.backbone import *
import my_ops
from my_ops import IPF, my_IPF
is_cuda = torch.cuda.is_available()
def to_var(x):
    if is_cuda:
        x = x.cuda()
    return x

CNN = eval(cfg.BACKBONE)


def lexico_iter(lex):
    return itertools.combinations(lex, 2)


def normalize_over_channels(x):
    channel_norms = torch.norm(x, dim=1, keepdim=True)
    return x / channel_norms


def concat_features(embeddings, num_vertices):
    res = torch.cat([embedding[:, :num_v] for embedding, num_v in zip(embeddings, num_vertices)], dim=-1)
    return res.transpose(0, 1)


def construct_hyperE(ori_graphs, batch_size, device):
    nmax = max([g.num_nodes for g in ori_graphs])
    emax = max([g.hyperedge_index.shape[1] for g in ori_graphs])
    hyperE = torch.zeros(batch_size, nmax, nmax, nmax, emax, device=device)
    for b, g in enumerate(ori_graphs):
        hyperE[b][g.hyperedge_index[0], g.hyperedge_index[1], g.hyperedge_index[2], torch.arange(
            g.hyperedge_index.shape[1])] = 1
    return hyperE, nmax, emax


def hyperedge_affinity(attrs1, attrs2):
    ret_list = []
    for attr1, attr2 in zip(attrs1, attrs2):
        X = attr1.unsqueeze(1).expand(attr1.shape[0], attr2.shape[0], attr1.shape[1])
        Y = attr2.unsqueeze(0).expand(attr1.shape[0], attr2.shape[0], attr2.shape[1])
        dist = torch.sum(torch.pow(X - Y, 2), dim=-1)
        dist[torch.isnan(dist)] = float("inf")
        ret_list.append(torch.exp(- dist / cfg.NGM.SIGMA3))
    return ret_list


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
            self.global_state_dim, self.build_edge_features_from_node_features.num_edge_features)
        #self.hyperedge_affinity = InnerProductWithWeightsAffinity(
        #    self.global_state_dim, self.build_edge_features_from_node_features.num_edge_features)

        self.rescale = cfg.PROBLEM.RESCALE
        self.tau = cfg.NGM.SK_TAU
        self.univ_size = cfg.NGM.UNIV_SIZE

        self.sinkhorn = Sinkhorn(max_iter=cfg.NGM.SK_ITER_NUM, tau=self.tau, epsilon=cfg.NGM.SK_EPSILON)
        self.sinkhorn_mgm = Sinkhorn(max_iter=cfg.NGM.SK_ITER_NUM, epsilon=cfg.NGM.SK_EPSILON, tau=1.)
        self.gnn_layer = cfg.NGM.GNN_LAYER
        for i in range(self.gnn_layer):
            if i == 0:
                gnn_layer = HyperGNNLayer(
                    1, 1, cfg.NGM.GNN_FEAT[i] + cfg.NGM.SK_EMB, cfg.NGM.GNN_FEAT[i],
                    sk_channel=cfg.NGM.SK_EMB, sk_tau=self.tau
                )
            else:
                gnn_layer = HyperGNNLayer(
                    cfg.NGM.GNN_FEAT[i - 1] + cfg.NGM.SK_EMB, cfg.NGM.GNN_FEAT[i - 1],
                    cfg.NGM.GNN_FEAT[i] + cfg.NGM.SK_EMB, cfg.NGM.GNN_FEAT[i],
                    sk_channel=cfg.NGM.SK_EMB, sk_tau=self.tau
                )
            self.add_module('gnn_layer_{}'.format(i), gnn_layer)
        self.classifier = nn.Linear(cfg.NGM.GNN_FEAT[-1] + cfg.NGM.SK_EMB, 1)

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
            orig_graph = self.build_edge_features_from_node_features(graph, hyperedge=True)
            orig_graph_list.append(orig_graph)

        global_weights_list = [
            torch.cat([global_src, global_tgt], axis=-1) for global_src, global_tgt in lexico_iter(global_list)
        ]

        global_weights_list = [normalize_over_channels(g) for g in global_weights_list]

        unary_affs_list = [
            self.vertex_affinity([item.x for item in g_1], [item.x for item in g_2], global_weights)
            for (g_1, g_2), global_weights in zip(lexico_iter(orig_graph_list), global_weights_list)
        ]

        quadratic_affs_list = [
            self.edge_affinity([item.edge_attr for item in g_1], [item.edge_attr for item in g_2], global_weights)
            for (g_1, g_2), global_weights in zip(lexico_iter(orig_graph_list), global_weights_list)
        ]

        quadratic_affs_list = [[0.5 * x for x in quadratic_affs] for quadratic_affs in quadratic_affs_list]

        order3_affs_list = [
            hyperedge_affinity([item.hyperedge_attr for item in g_1], [item.hyperedge_attr for item in g_2])
            for (g_1, g_2), global_weights in zip(lexico_iter(orig_graph_list), global_weights_list)
        ]

        s_list, mgm_s_list, x_list, mgm_x_list, indices = [], [], [], [], []

        for unary_affs, quadratic_affs, order3_affs, (g1, g2), (idx1, idx2) in \
            zip(unary_affs_list, quadratic_affs_list, order3_affs_list, lexico_iter(orig_graph_list), lexico_iter(range(num_graphs))):
            kro_G, kro_H = data_dict['KGHs'] if num_graphs == 2 else data_dict['KGHs']['{},{}'.format(idx1, idx2)]
            Kp = torch.stack(pad_tensor(unary_affs), dim=0)
            Ke = torch.stack(pad_tensor(quadratic_affs), dim=0)
            He = torch.stack(pad_tensor(order3_affs))
            K = construct_aff_mat(Ke, Kp, kro_G, kro_H)

            # build hyper graph tensor H
            hyperE1, nmax1, emax1 = construct_hyperE(g1, batch_size, He.device)
            hyperE2, nmax2, emax2 = construct_hyperE(g2, batch_size, He.device)
            H = torch.bmm(torch.bmm(
                hyperE1.reshape(batch_size, -1, emax1), He), hyperE2.reshape(batch_size, -1, emax2).transpose(1, 2))\
                .reshape(batch_size, nmax1, nmax1, nmax1, nmax2, nmax2, nmax2).permute(0, 4, 1, 5, 2, 6, 3)\
                .reshape(batch_size, nmax1*nmax2, nmax1*nmax2, nmax1*nmax2)

            if num_graphs == 2: data_dict['aff_mat'] = K

            if cfg.NGM.FIRST_ORDER:
                emb = Kp.transpose(1, 2).contiguous().view(Kp.shape[0], -1, 1)
            else:
                emb = torch.ones(K.shape[0], K.shape[1], 1, device=K.device)

            if cfg.NGM.POSITIVE_EDGES:
                adjs = [(K > 0).to(K.dtype), (H > 0).to(H.dtype)]
            else:
                adjs = [(K != 0).to(K.dtype), (H != 0).to(H.dtype)]

            emb_edges = [K.unsqueeze(-1), to_sparse(H.unsqueeze(-1), dense_dim=2)]

            # NGM qap solver
            for i in range(self.gnn_layer):
                gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
                emb_edges, emb = gnn_layer(adjs, emb_edges, emb, n_points[idx1], n_points[idx2]) #, weight=[1, 0.1])

            v = self.classifier(emb)

            if cfg.OPTIMIZATION_METHOD == 'Direct' and cfg.normalize_gnn_outputs_channles:
                v = normalize_over_channels(v)

            s = v.view(v.shape[0], points[idx2].shape[1], -1).transpose(1, 2)
            '''
            if cfg.OPTIMIZATION_METHOD == 'Sinkhorn':
                ss = self.sinkhorn(s, n_points[idx1], n_points[idx2], dummy_row=True)
            else:
                ss = s
            '''

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
                                    avg_diff = calc_differences_from_marginals(
                                        softs[b, :effective_rows, :effective_cols], effective_rows, row_marginals,
                                        effective_cols, column_marginals)

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
                                    softs = self.sinkhorn(s, n_points[idx1], n_points[idx2], dummy_row=True)

                                if cfg.prediction_type == 'row_normalization':
                                    softs = torch.zeros_like(s)
                                    for b in range(s.shape[0]):
                                        effective_rows = n_points[idx1][b]
                                        effective_cols = n_points[idx2][b]
                                        mat = torch.sigmoid(
                                            s[b, :effective_rows, :effective_cols].view(-1, effective_rows,
                                                                                        effective_cols))
                                        softs[b, :effective_rows, :effective_cols] = mat / mat.sum(dim=2, keepdim=True)

                                if cfg.prediction_type == 'dustbin':
                                    softs = torch.zeros_like(s)
                                    for b in range(s.shape[0]):
                                        effective_rows = n_points[idx1][b]
                                        effective_cols = n_points[idx2][b]
                                        # Run the optimal transport.
                                        mat = s[b, :effective_rows, :effective_cols].view(-1, effective_rows,
                                                                                          effective_cols)
                                        scores = log_optimal_transport(mat, self.bin_score, iters=25)
                                        ss = torch.sigmoid(scores)
                                        softs[b, :effective_rows, :effective_cols] = ss[:, :-1, :-1]

                                if cfg.prediction_type == 'gt_marginals':
                                    for b in range(s.shape[0]):
                                        effective_rows = n_points[idx1][b]
                                        effective_cols = n_points[idx2][b]
                                        mat_to_ipf = s[b, :effective_rows, :effective_cols]
                                        gt_premarginal_assignments = gt_perm_mats[0][b, :effective_rows,
                                                                     :effective_cols]
                                        row_marginals = torch.sum(gt_premarginal_assignments, dim=1)
                                        epsilon_row_marginals = torch.where(row_marginals > 0, row_marginals, to_var(
                                            torch.tensor(1e-7, dtype=torch.float)))
                                        column_marginals = torch.sum(gt_premarginal_assignments, dim=0)
                                        epsilon_column_marginals = torch.where(column_marginals > 0, column_marginals,
                                                                               to_var(torch.tensor(1e-7,
                                                                                                   dtype=torch.float)))

                                        softs[b, :effective_rows, :effective_cols] = my_IPF(torch.sigmoid(mat_to_ipf),
                                                                                            epsilon_row_marginals,
                                                                                            epsilon_column_marginals)

                    else:
                        softs = self.sinkhorn(s, n_points[idx1], n_points[idx2], dummy_row=True)
                        if cfg.calc_MAE:
                            for b in range(s.shape[0]):
                                effective_rows = n_points[idx1][b]
                                effective_cols = n_points[idx2][b]
                                x_prenorm = hungarian(s[b, :effective_rows, :effective_cols])
                                row_marginals = torch.sum(x_prenorm, dim=1)
                                column_marginals = torch.sum(x_prenorm, dim=0)
                                avg_diff = calc_differences_from_marginals(softs[b, :effective_rows, :effective_cols],
                                                                               effective_rows, row_marginals,
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
                    effective_rows = n_points[idx1][b]
                    effective_cols = n_points[idx2][b]
                    x_prenorm = hungarian(s[b, :effective_rows, :effective_cols])
                    row_marginals = torch.sum(x_prenorm, dim=1)
                    column_marginals = torch.sum(x_prenorm, dim=0)

                    # Run the optimal transport.
                    mat = s[b, :effective_rows, :effective_cols].view(-1, effective_rows, effective_cols)
                    scores = log_optimal_transport(mat, self.bin_score, iters=25)
                    ss = torch.exp(scores)
                    softs[b, :effective_rows, :effective_cols] = ss[:, :-1, :-1]
                    if cfg.calc_MAE:
                        avg_diff = calc_differences_from_marginals(softs[b, :effective_rows, :effective_cols],
                                                                   effective_rows, row_marginals, effective_cols,
                                                                   column_marginals)

            x = hungarian(softs, n_points[idx1], n_points[idx2])
            s_list.append(softs)
            x_list.append(x)
            indices.append((idx1, idx2))

        if num_graphs > 2 and cfg.PROBLEM.TYPE == '2GM':
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
                    #e, v = torch.symeig(joint_S[b], eigenvectors=True)
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
                    if cfg.OPTIMIZATION_METHOD == 'Sinkhorn' or cfg.MATCHING_TYPE == 'Balanced':
                        s = self.sinkhorn_mgm(torch.log(torch.relu(s)), n_points[idx1], n_points[idx2]) # only perform row/col norm, do not perform exp

                    x = hungarian(s, n_points[idx1], n_points[idx2])

                    mgm_s_list.append(s)
                    mgm_x_list.append(x)

        elif cfg.PROBLEM.TYPE == 'MGM' and cfg.OPTIMIZATION_METHOD == 'Direct':
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
