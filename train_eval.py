from torch.autograd import Variable

import torch.optim as optim
import time
import xlwt
from datetime import datetime
from pathlib import Path
from tensorboardX import SummaryWriter
import torch
import os
import numpy

from src.dataset.data_loader import GMDataset, get_dataloader
from models.GMN.displacement_layer import Displacement
from src.loss_func import *
from src.evaluation_metric import matching_accuracy
from src.parallel import DataParallel
from src.utils.model_sl import load_model, save_model
from eval import eval_model
from src.utils.data_to_cuda import data_to_cuda
from src.lap_solvers.hungarian import hungarian



from src.utils.config import cfg
import my_ops

is_cuda = torch.cuda.is_available()
def to_var(x):
    if is_cuda:
        x = x.cuda()
    return x

def build_softmatching_loss(matching_permutation_matrix,pos_weight):
    dim, m, n= matching_permutation_matrix.size()
    matrix_residuals = torch.zeros_like(matching_permutation_matrix,dtype=torch.float)

    if cfg.MATCHING_TYPE == 'Balanced':
        col_sums = torch.sum(matching_permutation_matrix, dim=1)
        col_residuals = torch.abs(torch.ones_like(col_sums) - col_sums) / n
        for d in range(dim):
            for r in range(m):
                matrix_residuals[d][r] = matrix_residuals[d][r] + col_residuals[d]

        return pos_weight* matrix_residuals


    elif cfg.MATCHING_TYPE == 'Unbalanced':
        col_sums = torch.sum(matching_permutation_matrix, dim=1)
        col_residuals = col_sums - torch.ones_like(col_sums)
        col_residuals = torch.clamp(col_residuals, min=0.0) / n
        for d in range(dim):
            for r in range(m):
                matrix_residuals[d][r] = matrix_residuals[d][r] + col_residuals[d]

        return pos_weight* matrix_residuals



def build_wbce_loss(matching_permutation_matrix, matching_gt_tiled, pos_weight):
    loss = torch.nn.BCELoss(reduction='none')
    loss_output = loss(to_var(matching_permutation_matrix.float()), to_var(matching_gt_tiled.float()))
    loss_output += pos_weight * loss_output * to_var(matching_gt_tiled.float())
    return loss_output

def build_2step_cycles(focal_graph_pair, graph_indices, pred_perm_mats):

    ref_source_graph = focal_graph_pair[0]
    ref_target_graph = focal_graph_pair[1]

    graph_indices_transpose = [tuple(numpy.flip(ind)) for ind in graph_indices]
    pred_perm_mat_transpose = [torch.transpose(pred_perm_mat,1,2) for pred_perm_mat in pred_perm_mats]

    graph_cycleparts = [((g_ind_tup_a, g_ind_tup_b), torch.matmul(pred_perm_mat_a,pred_perm_mat_b)) for g_ind_tup_a,pred_perm_mat_a in zip(graph_indices,pred_perm_mats) for g_ind_tup_b,pred_perm_mat_b in zip(graph_indices,pred_perm_mats) if g_ind_tup_a[0] == ref_source_graph and g_ind_tup_b[1] == ref_target_graph and g_ind_tup_a[1] == g_ind_tup_b[0] and g_ind_tup_a[1] != ref_target_graph]
    graph_cycleparts_t_b = [((g_ind_tup_a, g_ind_tup_b),torch.matmul(pred_perm_mat_a,pred_perm_mat_b))  for g_ind_tup_a,pred_perm_mat_a in zip(graph_indices, pred_perm_mats) for g_ind_tup_b,pred_perm_mat_b in zip(graph_indices_transpose,pred_perm_mat_transpose) if g_ind_tup_a[0] == ref_source_graph and g_ind_tup_b[1] == ref_target_graph and g_ind_tup_a[1] == g_ind_tup_b[0] and g_ind_tup_a[1] != ref_target_graph]
    graph_cycleparts_t_a = [((g_ind_tup_a, g_ind_tup_b),torch.matmul(pred_perm_mat_a,pred_perm_mat_b)) for g_ind_tup_a, pred_perm_mat_a in zip(graph_indices_transpose,pred_perm_mat_transpose) for g_ind_tup_b, pred_perm_mat_b in zip(graph_indices,pred_perm_mats) if g_ind_tup_a[0] == ref_source_graph and g_ind_tup_b[1] == ref_target_graph and g_ind_tup_a[1] == g_ind_tup_b[0] and g_ind_tup_a[1] != ref_target_graph]

    graph_cycleparts_indices_tuples = [t[0] for t in graph_cycleparts] + [t[0] for t in graph_cycleparts_t_b] + [t[0] for t in graph_cycleparts_t_a]
    graph_cycleparts_preds = [t[1] for t in graph_cycleparts] + [t[1] for t in graph_cycleparts_t_b] + [t[1] for t in graph_cycleparts_t_a]

    return graph_cycleparts_indices_tuples, graph_cycleparts_preds

def build_cycle_consistency_loss(focal_perm_mat_pred, graph_cycleparts_preds, lagrange_multiplier):
    all_cycles_loss = to_var(torch.zeros(focal_perm_mat_pred.size()))
    for cyc in graph_cycleparts_preds:
        cycle_loss_c = lagrange_multiplier*torch.where(focal_perm_mat_pred < cyc, cyc, to_var(torch.zeros(1)))
        all_cycles_loss += cycle_loss_c

    return all_cycles_loss

def direction_encoder_gradient_calcuate_w_illust(log_alpha_w_noise, log_alpha_w_noise_permutation_matrix, train_wbce_loss, samples_per_num_train, n1_gt, n2_gt):
    with torch.no_grad():
        log_alpha_w_noise_w_e_theta = log_alpha_w_noise.clone()

        log_alpha_minus_noise_w_e_theta = log_alpha_w_noise.clone()  ###two sided

        reattempt = True
        while reattempt:
            # associate the perturbation to its correlated position in log_alpha_w_noise according to the ground truth permutation
            log_alpha_w_noise_w_e_theta += cfg.loss_epsilon * train_wbce_loss
            log_alpha_minus_noise_w_e_theta -= cfg.loss_epsilon * train_wbce_loss  ###two sided

            # Solve a matching problem for a batch of matrices.
            hungarian_matching_permutation_matrix_with_epsilon_theta = hungarian(log_alpha_w_noise_w_e_theta, n1_gt, n2_gt)
            hungarian_matching_permutation_matrix_minus_epsilon_theta = hungarian(log_alpha_minus_noise_w_e_theta, n1_gt, n2_gt) ###two sided

            #encoder_direction_matrix = (-1)*hungarian_matching_permutation_matrix_with_epsilon_theta + log_alpha_w_noise_permutation_matrix
            encoder_direction_matrix = (-1)*hungarian_matching_permutation_matrix_with_epsilon_theta + hungarian_matching_permutation_matrix_minus_epsilon_theta ###two sided
            encoder_direction_matrix = encoder_direction_matrix.type(torch.float)
            batch_size = log_alpha_w_noise.size()[0]
            if torch.all(torch.eq(encoder_direction_matrix, to_var(torch.zeros([batch_size, encoder_direction_matrix.size()[1], encoder_direction_matrix.size()[2]])))) and torch.sum(train_wbce_loss) > 0.:
                cfg.loss_epsilon *= 1.1

                print("*************************zero gradients loss positive")
                print("*********increasing epsilon by 10%")
                reattempt = False
            else:
                reattempt = False

        return encoder_direction_matrix


def train_eval_model(model,
                     criterion,
                     optimizer,
                     dataloader,
                     tfboard_writer,
                     num_epochs=25,
                     start_epoch=0,
                     xls_wb=None):
    print('Start training...')

    since = time.time()
    dataset_size = len(dataloader['train'].dataset)
    displacement = Displacement()

    device = next(model.parameters()).device
    print('model on device: {}'.format(device))

    alphas = torch.tensor(cfg.EVAL.PCK_ALPHAS, dtype=torch.float32, device=device)  # for evaluation
    checkpoint_path = Path(cfg.OUTPUT_PATH) / ('params'+'_'+str(cfg.MATCHING_TYPE) + '_' + str(cfg.source_partial_kpt_len)+'_'+str(cfg.target_partial_kpt_len)+'_GConvNorma_'+str(cfg.crossgraph_s_normalization)+str(cfg.OPTIMIZATION_METHOD)+'_sample_'+str(cfg.samples_per_num_train)+now_time+'_'+str(cfg.PROBLEM.TYPE))
    #checkpoint_path = Path(cfg.OUTPUT_PATH) / 'params'
    if not checkpoint_path.exists():
        checkpoint_path.mkdir(parents=True)

    model_path, optim_path = '',''
    if start_epoch > 0:
        model_path = str(checkpoint_path / 'params_{:04}.pt'.format(start_epoch))
        optim_path = str(checkpoint_path / 'optim_{:04}.pt'.format(start_epoch))
    if len(cfg.PRETRAINED_PATH) > 0:
        model_path = cfg.PRETRAINED_PATH
    if len(model_path) > 0:
        print('Loading model parameters from {}'.format(model_path))
        load_model(model, model_path, strict=False)
    if len(optim_path) > 0:
        print('Loading optimizer state from {}'.format(optim_path))
        optimizer.load_state_dict(torch.load(optim_path))

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=cfg.TRAIN.LR_STEP,
                                               gamma=cfg.TRAIN.LR_DECAY,
                                               last_epoch=cfg.TRAIN.START_EPOCH - 1)

    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        #print("trying sinkhorn anyhow!")

        print("sigma_noise= ", str(cfg.sigma_norm))

        if cfg.PROBLEM.TYPE in ['MGM']:
            if cfg.OPTIMIZATION_METHOD == 'Direct':
                if cfg.penalty_method_on_cycle:
                    cfg.lagrange_multiplier += cfg.penalty_epoch_increase
                    print("lagrange_multiplier= ", str(cfg.lagrange_multiplier))

        model.train()  # Set model to training mode

        print('lr = ' + ', '.join(['{:.2e}'.format(x['lr']) for x in optimizer.param_groups]))

        epoch_loss = 0.0
        running_loss = 0.0
        running_since = time.time()
        iter_num = 0

        # Iterate over data.
        for inputs in dataloader['train']:

            if model.module.device != torch.device('cpu'):
                inputs = data_to_cuda(inputs)

            iter_num = iter_num + 1

            # zero the parameter gradients
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                # forward
                outputs = model(inputs)

                if cfg.PROBLEM.TYPE == '2GM':
                    assert 'ds_mat' in outputs
                    assert 'perm_mat' in outputs
                    assert 'gt_perm_mat' in outputs


                    # compute loss
                    if cfg.TRAIN.LOSS_FUNC == 'offset':
                        d_gt, grad_mask = displacement(outputs['gt_perm_mat'], *outputs['Ps'], outputs['ns'][0])
                        d_pred, _ = displacement(outputs['ds_mat'], *outputs['Ps'], outputs['ns'][0])
                        loss = criterion(d_pred, d_gt, grad_mask)
                    elif cfg.TRAIN.LOSS_FUNC in ['perm', 'ce', 'hung']:

                        if cfg.OPTIMIZATION_METHOD =='Sinkhorn':
                            if cfg.train_noise_factor and outputs['ds_mat'].requires_grad:
                                if cfg.samples_per_num_train > 1:
                                    outputs['gt_perm_mat'] = outputs['gt_perm_mat'].repeat(cfg.samples_per_num_train, 1, 1)
                                    outputs['ns'][0] = outputs['ns'][0].repeat(cfg.samples_per_num_train)
                                    outputs['ns'][1] = outputs['ns'][1].repeat(cfg.samples_per_num_train)

                            loss = criterion(outputs['ds_mat'], outputs['gt_perm_mat'], *outputs['ns'])


                        if cfg.OPTIMIZATION_METHOD =='superglue':

                            loss = criterion(outputs['ds_mat'], outputs['gt_perm_mat'], *outputs['ns'])

                        elif cfg.OPTIMIZATION_METHOD == 'Direct':  # direct optimization
                            pos_weight = torch.tensor(cfg.pos_weight)

                            if cfg.train_noise_factor:
                                sigma_tmp = to_var(torch.ones([outputs['ds_mat'].size()[0], 1], dtype=torch.float)) / cfg.sigma_norm
                                outputs['ds_mat'], _ = my_ops.my_phi_and_gamma_sigma_unbalanced(outputs['ds_mat'], cfg.samples_per_num_train,
                                                                                     cfg.train_noise_factor,
                                                                                     sigma_tmp)

                                # Solve a matching problem for a batch of matrices, if noise is added.
                                # tiled variables, to compare to many permutations
                                if cfg.samples_per_num_train > 1:
                                    outputs['gt_perm_mat'] = outputs['gt_perm_mat'].repeat(cfg.samples_per_num_train, 1, 1)
                                    outputs['ns'][0] = outputs['ns'][0].repeat(cfg.samples_per_num_train)
                                    outputs['ns'][1] = outputs['ns'][1].repeat(cfg.samples_per_num_train)


                                outputs['perm_mat'] = hungarian(outputs['ds_mat'], outputs['ns'][0], outputs['ns'][1])

                            # calculate weighted bce loss without reduction
                            train_wbce_loss = build_wbce_loss(outputs['perm_mat'], outputs['gt_perm_mat'], pos_weight)


                            encoder_gradient_direction_matrix = direction_encoder_gradient_calcuate_w_illust(
                                outputs['ds_mat'], outputs['perm_mat'], train_wbce_loss, 1,  outputs['ns'][0], outputs['ns'][1])

                            # calculate loss to optimize encoder
                            encoder_gradient_direction_matrix = (1. / 1.) * encoder_gradient_direction_matrix

                            loss = torch.sum(outputs['ds_mat'] * to_var(encoder_gradient_direction_matrix))

                    elif cfg.TRAIN.LOSS_FUNC == 'hamming':
                        loss = criterion(outputs['perm_mat'], outputs['gt_perm_mat'])
                    elif cfg.TRAIN.LOSS_FUNC == 'plain':
                        loss = torch.sum(outputs['loss'])
                    else:
                        raise ValueError('Unsupported loss function {} for problem type {}'.format(cfg.TRAIN.LOSS_FUNC, cfg.PROBLEM.TYPE))

                    # compute accuracy
                    acc, _, __ = matching_accuracy(outputs['perm_mat'], outputs['gt_perm_mat'], outputs['ns'][0])


                elif cfg.PROBLEM.TYPE in ['MGM', 'MGMC']:
                    if not cfg.OPTIMIZATION_METHOD == "BBGM":
                        assert 'ds_mat_list' in outputs
                    assert 'graph_indices' in outputs
                    assert 'perm_mat_list' in outputs
                    assert 'gt_perm_mat_list' in outputs


                    # compute loss & accuracy
                    if cfg.TRAIN.LOSS_FUNC in ['perm', 'ce' 'hung']:
                        if cfg.OPTIMIZATION_METHOD == 'Sinkhorn':
                            loss = torch.zeros(1, device=model.module.device)
                            ns = outputs['ns']

                            for s_pred, x_gt, (idx_src, idx_tgt) in \
                                    zip(outputs['ds_mat_list'], outputs['gt_perm_mat_list'], outputs['graph_indices']):
                                l = criterion(s_pred, x_gt, ns[idx_src], ns[idx_tgt])
                                loss += l
                            loss /= len(outputs['ds_mat_list'])

                        elif cfg.OPTIMIZATION_METHOD == 'Direct':  # direct optimization
                            pos_weight = torch.tensor(cfg.pos_weight)

                            loss = torch.zeros(1, device=model.module.device)
                            ns = outputs['ns'] #number of sampled nodes, each is (#keypoints at source graph, #keypoints at target graph)

                            graph_cycleparts_preds_all = []
                            for s_pred, perm_mat_pred, x_gt, (idx_src, idx_tgt) in \
                                    zip(outputs['ds_mat_list'], outputs['perm_mat_list'], outputs['gt_perm_mat_list'], outputs['graph_indices']):

                                if cfg.train_noise_factor:
                                    sigma_tmp = to_var(torch.ones([s_pred.size()[0], 1],dtype=torch.float)) / cfg.sigma_norm
                                    s_pred, _ = my_ops.my_phi_and_gamma_sigma_unbalanced(s_pred, cfg.samples_per_num_train, cfg.train_noise_factor, sigma_tmp)

                                    if cfg.samples_per_num_train > 1:
                                        x_gt = x_gt.repeat(cfg.samples_per_num_train, 1, 1)
                                        ns_src = ns[idx_src].repeat(cfg.samples_per_num_train)
                                        ns_trg = ns[idx_tgt].repeat(cfg.samples_per_num_train)

                                    else:
                                        ns_src = ns[idx_src][0]
                                        ns_trg = ns[idx_tgt][0]

                                    perm_mat = hungarian(s_pred, ns_src, ns_trg)

                                #no noise situation
                                else:
                                    ns_src = ns[idx_src]
                                    ns_trg = ns[idx_tgt]
                                    perm_mat = perm_mat_pred

                                # calculate weighted bce loss without reduction
                                train_wbce_loss = build_wbce_loss(perm_mat, x_gt, pos_weight)

                                # calculate 2step cycle consistency loss without reduction

                                graph_cycleparts_tup_indices, graph_cycleparts_preds = build_2step_cycles((idx_src, idx_tgt), outputs['graph_indices'], outputs['perm_mat_list'])
                                for p in range(len(graph_cycleparts_preds)):
                                    graph_cycleparts_preds[p] = graph_cycleparts_preds[p].repeat(cfg.samples_per_num_train, 1, 1)

                                graph_cycleparts_preds_all.append(graph_cycleparts_preds[0])
                                cycle_consistency_loss = build_cycle_consistency_loss(perm_mat, graph_cycleparts_preds, cfg.lagrange_multiplier)

                                if cfg.PROBLEM.UNSUPERVISED:
                                    total_loss = cycle_consistency_loss
                                else:
                                    total_loss = cycle_consistency_loss + train_wbce_loss
 
                                encoder_gradient_direction_matrix = direction_encoder_gradient_calcuate_w_illust(s_pred, perm_mat, total_loss, 1, ns_src, ns_trg)

                                # calculate loss to optimize encoder
                                encoder_gradient_direction_matrix = (1. / 1.) * encoder_gradient_direction_matrix

                                l = torch.sum(s_pred * to_var(encoder_gradient_direction_matrix))
                                loss += l
                            loss /= len(outputs['ds_mat_list'])

                        '''
                        elif cfg.OPTIMIZATION_METHOD == 'Direct' and cfg.MATCHING_TYPE =='Balanced':  # direct optimization
                            pos_weight = torch.tensor(cfg.pos_weight)
                            loss = torch.zeros(1, device=model.module.device)
                            ns = outputs['ns']

                            for s_pred, x_gt, (idx_src, idx_tgt) in \
                                    zip(outputs['ds_mat_list'], outputs['gt_perm_mat_list'], outputs['graph_indices']):

                                if cfg.train_noise_factor:
                                    sigma_tmp = to_var(torch.ones([s_pred.size()[0], 1],dtype=torch.float)) / cfg.sigma_norm
                                    s_pred, _ = my_ops.my_phi_and_gamma_sigma_unbalanced(s_pred, cfg.samples_per_num_train, cfg.train_noise_factor, sigma_tmp)

                                if cfg.samples_per_num_train > 1:
                                    x_gt = x_gt.repeat(cfg.samples_per_num_train, 1, 1)
                                    ns_src = ns[idx_src].repeat(cfg.samples_per_num_train)
                                    ns_trg = ns[idx_tgt].repeat(cfg.samples_per_num_train)
                                else:
                                    ns_src = ns[idx_src]
                                    ns_trg = ns[idx_tgt]

                                perm_mat = hungarian(s_pred, ns_src, ns_trg)

                                # calculate weighted bce loss without reduction
                                train_wbce_loss = build_wbce_loss(perm_mat, x_gt, pos_weight)

                                encoder_gradient_direction_matrix = direction_encoder_gradient_calcuate_w_illust(
                                    s_pred, perm_mat, train_wbce_loss, 1, ns_src, ns_trg)

                                # calculate loss to optimize encoder
                                encoder_gradient_direction_matrix = (1. / 1.) * encoder_gradient_direction_matrix

                                l = torch.sum(s_pred * to_var(encoder_gradient_direction_matrix))
                                loss += l
                            loss /= len(outputs['ds_mat_list'])
                        '''
                    elif cfg.TRAIN.LOSS_FUNC == 'plain':
                        loss = torch.sum(outputs['loss'])
                    elif cfg.TRAIN.LOSS_FUNC == 'hamming':
                        ns = outputs['ns']

                        loss_i = 0
                        for i in range(len(outputs['perm_mat_list'])):
                            loss_i += criterion(outputs['perm_mat_list'][i], outputs['gt_perm_mat_list'][i])
                        loss = loss_i/ len(outputs['perm_mat_list'])
                    else:
                        raise ValueError('Unsupported loss function {} for problem type {}'.format(cfg.TRAIN.LOSS_FUNC, cfg.PROBLEM.TYPE))

                    # compute accuracy
                    acc = torch.zeros(1, device=model.module.device)
                    for x_pred, x_gt, (idx_src, idx_tgt) in \
                            zip(outputs['perm_mat_list'], outputs['gt_perm_mat_list'], outputs['graph_indices']):
                        a, _, __ = matching_accuracy(x_pred, x_gt, ns[idx_src])
                        if cfg.PROBLEM.TYPE in ['MGM', 'MGMC']:
                            acc += torch.mean(a)
                        else:
                            acc += torch.sum(a)
                    acc /= len(outputs['perm_mat_list'])

                    # compute cycle-consistency
                    if cfg.OPTIMIZATION_METHOD == "Direct":
                        if cfg.PROBLEM.TYPE in ['MGM', 'MGMC']:
                            cyc_const = torch.zeros(1, device=model.module.device)
                            for x_pred, x_2stepcycle, (idx_src, idx_tgt) in \
                                    zip(outputs['perm_mat_list'], graph_cycleparts_preds_all, outputs['graph_indices']):
                                c, _, __ = matching_accuracy(x_pred, x_2stepcycle, ns[idx_src])
                                cyc_const += torch.mean(c)

                            cyc_const /= len(outputs['perm_mat_list'])

                else:
                    raise ValueError('Unknown problem type {}'.format(cfg.PROBLEM.TYPE))

                # backward + optimize
                if cfg.FP16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                optimizer.step()

                batch_num = inputs['batch_size']

                # tfboard writer
                loss_dict = dict()
                loss_dict['loss'] = loss.item()
                tfboard_writer.add_scalars('loss', loss_dict, epoch * cfg.TRAIN.EPOCH_ITERS + iter_num)

                accdict = dict()
                accdict['matching accuracy'] = torch.mean(acc)
                tfboard_writer.add_scalars(
                    'training accuracy',
                    accdict,
                    epoch * cfg.TRAIN.EPOCH_ITERS + iter_num
                )
                if cfg.OPTIMIZATION_METHOD == "Direct":
                    if cfg.PROBLEM.TYPE in ['MGM', 'MGMC']:
                        cycle_consistency_dict = dict()
                        cycle_consistency_dict['cycle_consistency accuracy'] = torch.mean(cyc_const)
                        tfboard_writer.add_scalars(
                            'training cycle_consistency accuracy',
                            cycle_consistency_dict,
                            epoch * cfg.TRAIN.EPOCH_ITERS + iter_num
                        )

                # statistics
                running_loss += loss.item() * batch_num
                epoch_loss += loss.item() * batch_num

                if iter_num % cfg.STATISTIC_STEP == 0:
                    running_speed = cfg.STATISTIC_STEP * batch_num / (time.time() - running_since)
                    print('Epoch {:<4} Iteration {:<4} {:>4.2f}sample/s Loss={:<8.4f}'
                          .format(epoch, iter_num, running_speed, running_loss / cfg.STATISTIC_STEP / batch_num))
                    tfboard_writer.add_scalars(
                        'speed',
                        {'speed': running_speed},
                        epoch * cfg.TRAIN.EPOCH_ITERS + iter_num
                    )

                    tfboard_writer.add_scalars(
                        'learning rate',
                        {'lr_{}'.format(i): x['lr'] for i, x in enumerate(optimizer.param_groups)},
                        epoch * cfg.TRAIN.EPOCH_ITERS + iter_num
                    )

                    running_loss = 0.0
                    running_since = time.time()

        epoch_loss = epoch_loss / dataset_size

        save_model(model, str(checkpoint_path / 'params_{:04}.pt'.format(epoch + 1)))
        torch.save(optimizer.state_dict(), str(checkpoint_path / 'optim_{:04}.pt'.format(epoch + 1)))

        print('Epoch {:<4} Loss: {:.8f}'.format(epoch, epoch_loss))
        print()

        # Eval in each epoch
        accs = eval_model(model, alphas, dataloader['test'], xls_sheet=xls_wb.add_sheet('epoch{}'.format(epoch + 1)))
        acc_dict = {"{}".format(cls): single_acc for cls, single_acc in zip(dataloader['test'].dataset.classes, accs)}
        acc_dict['average'] = torch.mean(accs)
        tfboard_writer.add_scalars(
            'Eval acc',
            acc_dict,
            (epoch + 1) * cfg.TRAIN.EPOCH_ITERS
        )
        wb.save(wb.__save_path)

        scheduler.step()

        cfg.sigma_norm = cfg.sigma_norm*(1+cfg.sigma_decay)


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'
          .format(time_elapsed // 3600, (time_elapsed // 60) % 60, time_elapsed % 60))

    return model


if __name__ == '__main__':
    from src.utils.dup_stdout_manager import DupStdoutFileManager
    from src.utils.parse_args import parse_args
    from src.utils.print_easydict import print_easydict
    from src.utils.count_model_params import count_parameters

    args = parse_args('Deep learning of graph matching training & evaluation code.')

    import importlib
    mod = importlib.import_module(cfg.MODULE)
    Net = mod.Net

    torch.manual_seed(cfg.RANDOM_SEED)

    #cfg.PROBLEM.TYPE = 'MGM'
    dataset_len = {'train': cfg.TRAIN.EPOCH_ITERS * cfg.BATCH_SIZE, 'test': cfg.EVAL.SAMPLES}

    image_dataset = {
        x: GMDataset(cfg.DATASET_FULL_NAME,
                     sets=x,
                     problem=cfg.PROBLEM.TYPE,
                     length=dataset_len[x],
                     #cls=cfg.TRAIN.CLASS if x == 'train' else cfg.EVAL.CLASS,
                     obj_resize=cfg.PROBLEM.RESCALE)
        for x in ('train', 'test')}
    dataloader = {x: get_dataloader(image_dataset[x], fix_seed=(x == 'test'))
        for x in ('train', 'test')}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Net()
    '''
    Multiple_GPU_training = False
    
    if Multiple_GPU_training and torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = torch.nn.DataParallel(model)
        model.to(device)
    ##########
    else:
        model.to(device)
    '''

    model.to(device)
    if cfg.TRAIN.LOSS_FUNC.lower() == 'offset':
        criterion = RobustLoss(norm=cfg.TRAIN.RLOSS_NORM)
    elif cfg.TRAIN.LOSS_FUNC.lower() == 'perm':
        criterion = PermutationLoss()
    elif cfg.TRAIN.LOSS_FUNC.lower() == 'ce':
        criterion = CrossEntropyLoss()
    elif cfg.TRAIN.LOSS_FUNC.lower() == 'focal':
        criterion = FocalLoss(alpha=.5, gamma=0.)
    elif cfg.TRAIN.LOSS_FUNC.lower() == 'hung':
        criterion = PermutationLossHung()
    elif cfg.TRAIN.LOSS_FUNC.lower() == 'hamming':
        criterion = HammingLoss()
    else:
        raise ValueError('Unknown loss function {}'.format(cfg.TRAIN.LOSS_FUNC))


    if cfg.TRAIN.SEPARATE_BACKBONE_LR:
        '''
        if Multiple_GPU_training:
            backbone_ids = [id(item) for item in model.module.backbone_params]
            other_params = [param for param in model.parameters() if id(param) not in backbone_ids]

            model_params = [
                # {'params': other_params, 'lr': 1.5*cfg.TRAIN.LR, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY},
                {'params': other_params},
                {'params': model.module.backbone_params, 'lr': cfg.TRAIN.BACKBONE_LR}
            ]
        '''
        backbone_ids = [id(item) for item in model.backbone_params]
        other_params = [param for param in model.parameters() if id(param) not in backbone_ids]

        model_params = [
            #{'params': other_params, 'lr': 1.5*cfg.TRAIN.LR, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY},
            {'params': other_params},
            {'params': model.backbone_params, 'lr': cfg.TRAIN.BACKBONE_LR}
        ]
    else:
        model_params = model.parameters()

    if cfg.TRAIN.OPTIMIZER.lower() == 'sgd':
        optimizer = optim.SGD(model_params, lr=cfg.TRAIN.LR, momentum=cfg.TRAIN.MOMENTUM, nesterov=True)
    elif cfg.TRAIN.OPTIMIZER.lower() == 'adam':
        optimizer = optim.Adam(model_params, lr=cfg.TRAIN.LR)
    else:
        raise ValueError('Unknown optimizer {}'.format(cfg.TRAIN.OPTIMIZER))

    '''
    if cfg.OPTIMIZATION_METHOD == 'Direct':
        if cfg.TRAIN.OPTIMIZER.lower() == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=2*1e-4, momentum=cfg.TRAIN.MOMENTUM, nesterov=True)
        elif cfg.TRAIN.OPTIMIZER.lower() == 'adam':
            optimizer = optim.Adam(model_params, lr=cfg.TRAIN.LR)
            print("chose adam with lr", str(cfg.TRAIN.LR))
        '''
    if cfg.FP16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to enable FP16.")
        model, optimizer = amp.initialize(model, optimizer)

    model = DataParallel(model, device_ids=cfg.GPUS)

    if not Path(cfg.OUTPUT_PATH).exists():
        Path(cfg.OUTPUT_PATH).mkdir(parents=True)

    now_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    #tfboardwriter = SummaryWriter(logdir=str(Path(cfg.OUTPUT_PATH) / 'tensorboard' / 'training_{}'.format(now_time)))
    tfboardwriter = SummaryWriter(logdir=str(Path(cfg.OUTPUT_PATH) / ('tensorboard'+'_'+str(cfg.MATCHING_TYPE) +'_'+str(cfg.source_partial_kpt_len)+'_'+str(cfg.target_partial_kpt_len)+'_GConv_normalization_'+str(cfg.crossgraph_s_normalization)+str(cfg.OPTIMIZATION_METHOD)+'_sample_'+str(cfg.samples_per_num_train)+'_'+str(cfg.PROBLEM.TYPE)) / 'training_{}'.format(now_time)))

    log_path = Path(cfg.OUTPUT_PATH) / ('logs'+'_'+str(cfg.MATCHING_TYPE)+'_'+str(cfg.source_partial_kpt_len)+'_'+str(cfg.target_partial_kpt_len)+'_GConv_normalization_'+str(cfg.crossgraph_s_normalization)+str(cfg.OPTIMIZATION_METHOD)+'_sample_'+str(cfg.samples_per_num_train)+'_'+str(cfg.PROBLEM.TYPE))
    if not log_path.exists():
        log_path.mkdir(parents=True)

    wb = xlwt.Workbook()
    wb.__save_path = str(Path(cfg.OUTPUT_PATH) / ('train_eval_result_' + now_time + '.xls'))

    with DupStdoutFileManager(os.path.join(log_path, 'train_log_' + now_time + '.log')) as _:
        print_easydict(cfg)
        print('Number of parameters: {:.2f}M'.format(count_parameters(model) / 1e6))
        model = train_eval_model(model, criterion, optimizer, dataloader, tfboardwriter,
                                 #num_epochs=10,
                                 num_epochs=cfg.TRAIN.NUM_EPOCHS,
                                 start_epoch=cfg.TRAIN.START_EPOCH,
                                 xls_wb=wb)

    wb.save(wb.__save_path)
