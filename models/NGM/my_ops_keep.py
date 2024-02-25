
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import kendalltau
import torch
#https://github.com/src-d/lapjv
from lapjv import lapjv
from lap import lapjv as lapjv_unbalanced #supports unbalanced problems
import matplotlib.pyplot as plt

is_cuda = torch.cuda.is_available()

def to_var(x):
    if is_cuda:
        x = x.cuda()
    return x

def my_sample_uniform_and_order(n_lists, n_numbers, prob_inc, min_range, max_range):
    """Samples uniform random numbers, return sorted lists and the indices of their original values

    Returns a 2-D tensor of n_lists lists of n_numbers sorted numbers in the [0,1]
    interval, each of them having n_numbers elements.
    Lists are increasing with probability prob_inc.
    It does so by first sampling uniform random numbers, and then sorting them.
    Therefore, sorted numbers follow the distribution of the order statistics of
    a uniform distribution.
    It also returns the random numbers and the lists of permutations p such
    p(sorted) = random.
    Notice that if one ones to build sorted numbers in different intervals, one
    might just want to re-scaled this canonical form.

    Args:
    n_lists: An int,the number of lists to be sorted.
    n_numbers: An int, the number of elements in the permutation.
    prob_inc: A float, the probability that a list of numbers will be sorted in
    increasing order.

    Returns:
    ordered: a 2-D float tensor with shape = [n_list, n_numbers] of sorted lists
     of numbers.
    random: a 2-D float tensor with shape = [n_list, n_numbers] of uniform random
     numbers.
    permutations: a 2-D int tensor with shape = [n_list, n_numbers], row i
     satisfies ordered[i, permutations[i]) = random[i,:].

    """
    # sample n_lists samples from Bernoulli with probability of prob_inc
    random =(torch.empty(n_lists, n_numbers).uniform_(min_range, max_range))
    random = random.type(torch.float32)

    ordered, permutations = torch.sort(random, descending=True)

    return (ordered, random, permutations)

def my_sinkhorn_dummy(log_alpha, epsilon = 1e-3, iters = 20):
    """Performs incomplete Sinkhorn normalization to log_alpha.
    By a theorem by Sinkhorn and Knopp [1], a sufficiently well-behaved  matrix
    with positive entries can be turned into a doubly-stochastic matrix
    (i.e. its rows and columns add up to one) via the successive row and column
    normalization.
    -To ensure positivity, the effective input to sinkhorn has to be
    exp(log_alpha) (element wise).
    log_alpha = torch.exp(log_alpha)
    [1] Sinkhorn, Richard and Knopp, Paul.
    Concerning nonnegative matrices and doubly stochastic
    matrices. Pacific Journal of Mathematics, 1967
    Args:
    log_alpha: a 2D tensor of shape [n_rows, n_cols], n_rows<=n_cols
    n_iters: number of sinkhorn iterations (in practice, as little as 20
      iterations are needed to achieve decent convergence for N~100)
    Returns:
    A 3D tensor of close-to-partially-doubly-stochastic matrices (2D tensors are
      converted to 3D tensors with batch_size equals to 1)
    """
    batch_size = log_alpha.size()[0]
    n_rows = log_alpha.size()[1]
    n_cols = log_alpha.size()[2]
    log_alpha = log_alpha.view(-1, n_rows, n_cols)

    #add dummy rows in case of unbalanced matching problem in order to perform sinkhorn normalization
    if n_cols > n_rows:
        dummy_shape = list(log_alpha.shape)
        dummy_shape[1] = n_cols - n_rows
        s = torch.cat((log_alpha, to_var(torch.full(dummy_shape, 0.))), dim=1)
        new_n_rows = n_cols
        for b in range(batch_size):
            s[b, n_rows:new_n_rows, :n_cols] = epsilon

        log_alpha = s #extended with dummy rows


    '''
    plt.imshow(log_alpha[0].detach().numpy(), cmap=plt.cm.Blues)
    plt.colorbar()
    plt.title("mu representation before normalization with dummy rows ")
    plt.savefig("before.png")
    plt.show()
    '''
    for i in range(iters):
        #Returns the log of summed exponentials of each row of the input tensor in the given dimension dim
        #normalize on rows
        #log_alpha = log_alpha / torch.sum(log_alpha, dim=2, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=2, keepdim=True)

        #normalize on columns
        #log_alpha = log_alpha / torch.sum(log_alpha, dim=1, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=1, keepdim=True)

    # exponentiate
    log_alpha = torch.exp(log_alpha)

    partial_log_alpha = log_alpha[:, :n_rows, :]

    if torch.min(partial_log_alpha) < 0 or torch.max(partial_log_alpha) > 1:
        print(partial_log_alpha)
        print("partial_log_alpha error")
    if torch.isnan(partial_log_alpha).any():
        print(partial_log_alpha)
        print("partial_log_alpha error")
    return partial_log_alpha


def my_sample_uniform_sets(n_lists, n_numbers_in, n_numbers_inout, prob_inc, min_range, max_range):
    """Samples two sets of uniform random numbers, one set is contained in the other

    Returns a two 2D tensors of n_lists lists (batch size) of n_numbers_in and n_numbers_inout , such that the
    set of samples in n_numbers_in is contained in the set of samples of n_numbers_inout
    Args:
    n_lists: An int,the number of lists to be sorted.
    n_numbers_in: An int, the number of inlier elements of the contained set.
    n_numbers_inout: An int, the number of inlier + outlier elements of the containing set.
    prob_inc: A float, the probability that a list of numbers will be sorted in
    increasing order.

    Returns:
    sets: two 2D float tensors, one with shape = [n_lists, n_numbers_in[ and the other
     with shape = [n_list, n_numbers_inout] of uniform random numbers. The first set contains inlier samples,
     and is contained within the second with may contain outlier samples in addition.
    matching_ permutation: a 2D int tensor with shape = [n_list, n_numbers_in, n_numbers_inout]
     which is a (partial) permutation matrix denoting the matching between an element
     from the first set to the second set.

    """
    # sample n_lists samples for the contained set from Bernoulli with probability of prob_inc
    random_in = (torch.empty(n_lists, n_numbers_in).uniform_(min_range, max_range))
    random_in = random_in.type(torch.float32)

    # sample n_lists samples for the contained set from Bernoulli with probability of prob_inc
    if n_numbers_inout - n_numbers_in > 0:
        random_out = (torch.empty(n_lists, n_numbers_inout - n_numbers_in).uniform_(min_range, max_range))
        random_out = random_out.type(torch.float32)

        #concatenate outliers to inliers
        random_inout_cat = torch.cat((random_in, random_out), 1)
        random_inout = random_inout_cat.clone()
    else:
        random_inout = random_in.clone()

    # randomly permute the containing set
    for i in range(n_lists):
        rand_perm_idx = torch.randperm(n_numbers_inout)
        random_inout[i] = random_inout[i][rand_perm_idx]

    #build outer square differences partial matrix,
    # towards finding the ground truth partial permutation matrix
    #K dimensions are [batch_size, n_numbers_in, n_numbers_inout]
    K = (random_inout.unsqueeze(2) - random_in.unsqueeze(1))**2
    K = K.permute(0,2,1)

    # gt_matching_col_idx has shape [batch_size, n_numbers_in] specifying in each batch
    # to which column is row assigned
    gt_matching_col_idx = hungarian_matching(-K)
    gt_matching_matrix = my_listperm2matperm_unbalanced(gt_matching_col_idx, n_numbers_in, n_numbers_inout)

    return ((random_in, random_inout), gt_matching_matrix)

def my_sample_gumbel(shape):
    """Samples arbitrary-shaped standard gumbel variables.
    Args:
    shape: list of integers
    eps: float, for numerical stability

    Returns:
    A sample of standard Gumbel random variables
    """
    # Sample from Gumbel with expectancy 0 and variance
    beta = np.sqrt(0.2/(np.square(np.pi)))

    # Sample from Gumbel with expectancy 0 and variance 2
    #beta = np.sqrt(12./(np.square(np.pi)))

    # Sample from Gumbel with expectancy 0 and variance 3
    #beta = np.sqrt(18./(np.square(np.pi)))

    mu = -beta*np.euler_gamma

    U = np.random.gumbel(loc=mu, scale=beta, size=shape)
    return torch.from_numpy(U).float()

def my_phi_and_gamma_sigma_unbalanced(log_alpha, samples_per_num, noise_factor, sigma):
    """
    Args:
    log_alpha: 2D tensor (a matrix of shape [N, M])
      or 3D tensor (a batch of matrices of shape = [batch_size, N, M])

    Returns:
    log_alpha_w_noise: a 4D tensor of [batch_size, n_samples, N, M] of
          noisy samples of log_alpha, If n_samples = 1 then the output is 3D.
    """
    n = log_alpha.size()[1]
    m = log_alpha.size()[2]
    log_alpha = log_alpha.view(-1, n, m)
    batch_size = log_alpha.size()[0]

    if samples_per_num > 1:
        log_alpha_tiled = log_alpha.repeat(samples_per_num, 1, 1)
    else:
        log_alpha_tiled = log_alpha

    noise_sigma_tiled = to_var(torch.zeros((batch_size * samples_per_num, n, m)))
    if noise_factor == True:
        noise = to_var(my_sample_gumbel([batch_size * samples_per_num, n, m]))

        # rescale noise according to sigma
        sigma_tiled = sigma.repeat(samples_per_num, 1)
        for bm in range(batch_size * samples_per_num):
            noise_sigma_tiled[bm] = sigma_tiled[bm] * noise[bm]

        log_alpha_w_noise = log_alpha_tiled + noise_sigma_tiled

    else:
        log_alpha_w_noise = log_alpha_tiled

    return log_alpha_w_noise, noise_sigma_tiled


def my_sample_permutations(n_permutations, n_objects):
    """Samples a batch permutations from the uniform distribution.

    Returns a sample of n_permutations permutations of n_objects indices.
    Permutations are assumed to be represented as lists of integers
    (see 'listperm2matperm' and 'matperm2listperm' for conversion to alternative
    matricial representation). It does so by sampling from a continuous
    distribution and then ranking the elements. By symmetry, the resulting
    distribution over permutations must be uniform.

    Args:
    n_permutations: An int, the number of permutations to sample.
    n_objects: An int, the number of elements in the permutation.
      the embedding sources.

    Returns:
    A 2D integer tensor with shape [n_permutations, n_objects], where each
      row is a permutation of range(n_objects)

    """
    random_pre_perm = torch.empty(n_permutations, n_objects).uniform_(0, 1)
    _, permutations = torch.topk(random_pre_perm, k = n_objects)
    return permutations

def my_permute_batch_split(batch_split, permutations):
    """Scrambles a batch of objects according to permutations.

    It takes a 3D tensor [batch_size, n_objects, object_size]
    and permutes items in axis=1 according to the 2D integer tensor
    permutations, (with shape [batch_size, n_objects]) a list of permutations
    expressed as lists. For many dimensional-objects (e.g. images), objects have
    to be flattened so they will respect the 3D format, i.e. tf.reshape(
    batch_split, [batch_size, n_objects, -1])

    Args:
    batch_split: 3D tensor with shape = [batch_size, n_objects, object_size] of
      splitted objects
    permutations: a 2D integer tensor with shape = [batch_size, n_objects] of
      permutations, so that permutations[n] is a permutation of range(n_objects)

    Returns:
    A 3D tensor perm_batch_split with the same shape as batch_split,
      so that perm_batch_split[n, j,:] = batch_split[n, perm[n,j],:]

    """
    batch_size= permutations.size()[0]
    n_objects = permutations.size()[1]

    permutations = permutations.view(batch_size, n_objects, -1)
    perm_batch_split = torch.gather(batch_split, 1, permutations)
    return perm_batch_split


def my_listperm2matperm_unbalanced(listperm, n_rows, n_cols):
    """Converts a batch of permutations to its matricial form.

    Args:
    listperm: 2D tensor of permutations of shape [batch_size, n_objects] so that
      listperm[n] is a permutation of range(n_objects).

    Returns:
    a 3D tensor of permutations matperm of
      shape = [batch_size, n_objects, m_objects] so that matperm[n, :, :] is a
      permutation of the identity matrix, with matperm[n, i, listperm[n,i]] = 1
    """

    eye = np.eye(n_cols, n_cols)[listperm]
    eye = torch.tensor(eye, dtype=torch.int32)
    return eye

def my_matperm2listperm(matperm):
    """Converts a batch of permutations to its enumeration (list) form.

    Args:
    matperm: a 3D tensor of permutations of
      shape = [batch_size, n_objects, n_objects] so that matperm[n, :, :] is a
      permutation of the identity matrix. If the input is 2D, it is reshaped
      to 3D with batch_size = 1.
    dtype: output_type (tf.int32, tf.int64)

    Returns:
    A 2D tensor of permutations listperm, where listperm[n,i]
    is the index of the only non-zero entry in matperm[n, i, :]
    """
    batch_size = matperm.size()[0]
    n_objects = matperm.size()[1]
    matperm = matperm.view(-1, n_objects, n_objects)

    #_, argmax = matperm.max(-1)
    #argmax is the index location of each maximum value found(argmax)
    _, argmax = torch.max(matperm, dim=2, keepdim= True)
    argmax = argmax.view(batch_size, n_objects)
    return argmax

def my_invert_listperm_unbalanced(listperm, n_rows, n_cols):
    """Inverts a batch of permutations.

    Args:
    listperm: a 2D integer tensor of permutations listperm of
      shape = [batch_size, n_objects] so that listperm[n] is a permutation of
      range(n_objects)
    Returns:
    A 2D tensor of permutations listperm, where listperm[n,i]
    is the index of the only non-zero entry in matperm[n, i, :]
    """
    return my_matperm2listperm(torch.transpose(my_listperm2matperm_unbalanced(listperm, n_rows, n_cols), 1, 2))

def matrix_rep_trim(matrix_batch, ns_src, ns_trg):
    """Trims a batch of matrices to the shapes of n2_src, ns_trg

      Args:
        matrix_batch: A 3D tensor (a batch of matrices) with
          shape = [batch_size, N, N]. If 2D, the input is reshaped to 3D with
          batch_size = 1.
        nS_src: a list of number of source nodes
        ns_trg: a list of number of target nodes


      Returns:
        matrix_batch trimmed
      """
    batch_size = matrix_batch.size()[0]
    for b in range(batch_size):
        print(matrix_batch[b].size())
        matrix_batch[b] = matrix_batch[b,:ns_src[b], :ns_trg[b]]

    return matrix_batch



def hungarian_matching(matrix_batch):
  """Solves a matching problem for a batch of matrices.

  This is a wrapper for the scipy.optimize.linear_sum_assignment function. It
  solves the optimization problem max_P sum_i,j M_i,j P_i,j with P a
  permutation matrix. Notice the negative sign; the reason, the original
  function solves a minimization problem

  Args:
    matrix_batch: A 3D tensor (a batch of matrices) with
      shape = [batch_size, N, N]. If 2D, the input is reshaped to 3D with
      batch_size = 1.

  Returns:
    listperms, a 2D integer tensor of permutations with shape [batch_size, N]
      so that listperms[n, :] is the permutation of range(N) that solves the
      problem  max_P sum_i,j M_i,j P_i,j with M = matrix_batch[n, :, :].
  """

  def hungarian(x):
    if x.ndim == 2:
      x = np.reshape(x, [1, x.shape[0], x.shape[1]])
    #sol = np.zeros((x.shape[0], x.shape[1]), dtype=np.int32)
    lap_sol = np.zeros((x.shape[0], x.shape[1]), dtype=np.int32)
    for i in range(x.shape[0]):

        #for square problems:
        #sol[i, :] = linear_sum_assignment(-x[i, :])[1].astype(np.int32) #slower, deprecated
        #lap_sol[i, :]= lapjv(-x[i, :])[0].astype(np.int32)

        #for non square problems:
        lap_sol[i, :] = lapjv_unbalanced(-x[i, :], extend_cost=True)[1].astype(np.int32)

    return lap_sol

  lap_listperms = hungarian(matrix_batch.detach().cpu().numpy())
  lap_listperms = torch.from_numpy(lap_listperms)
  return lap_listperms

def my_kendall_tau(batch_perm1, batch_perm2):
  """Wraps scipy.stats kendalltau function.
  Args:
    batch_perm1: A 2D tensor (a batch of matrices) with
      shape = [batch_size, N]
    batch_perm2: same as batch_perm1

  Returns:
    A list of Kendall distances between each of the elements of the batch.
  """
  def kendalltau_batch(x, y):

    if x.ndim == 1:
      x = np.reshape(x, [1, x.shape[0]])
    if y.ndim == 1:
      y = np.reshape(y, [1, y.shape[0]])
    kendall = np.zeros((x.shape[0], 1), dtype=np.float32)
    for i in range(x.shape[0]):
      kendall[i, :] = kendalltau(x[i, :], y[i, :])[0]
    return kendall

  listkendall = kendalltau_batch(batch_perm1.numpy(), batch_perm2.numpy())
  listkendall = torch.from_numpy(listkendall)
  return listkendall


