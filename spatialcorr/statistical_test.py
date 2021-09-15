import matplotlib as mpl  
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
import math
import importlib
import seaborn as sns
from collections import defaultdict
from multiprocessing import Process, Queue, Manager
from sklearn.metrics.pairwise import euclidean_distances

import matplotlib.cm
from matplotlib.colors import ListedColormap

def build_normal_log_pdf(mean, cov, cov_inv=None):
    if cov_inv is None:
        cov_inv = np.linalg.inv(cov)
    cov_det = np.abs(np.linalg.det(cov))
    const = ((-len(mean)/2) * np.log(2*np.pi)) \
        - (0.5 * np.log(cov_det))
    def normal_log_pdf(x):
        cent = x - mean
        return const - (0.5 * np.dot(cent, np.dot(cov_inv, cent)))
    return normal_log_pdf


def kernel_estimation(
        kernel_matrix,
        X,
        verbose=10
    ):
    """
    X
        GxN matrix where G is number of genes and N is number of spots
    """
    # For simplicity
    K = kernel_matrix

    # Means will be a NxG matrix of the spot-wise
    # kernel-computed means
    means = (np.matmul(K, X.T).T / np.sum(K, axis=1)).T

    # Compute the difference of each spotwise expression 
    # from its mean
    devs = X.T-means
    
    # For each spot, compute the outer product of the deviation
    # vector with itself. These correspond to each term in the
    # summation used to calculate dynamic covariance at each spot.
    O = np.array([np.outer(d,d) for d in devs])

    # This is a bit complicated. For a given spot i, to compute 
    # its covariance matrix, we need to compute a weighted sum 
    # of these matrices:
    #
    #   \sum_j K_{ij} (X_j - mean_j)(X_j - mean_j)^T
    #
    # O is a list of matrices: 
    #       
    # (X_1 - mean_1)(X_1 - mean_1)^T ... (X_n - mean_n)(X_n - mean_n)^T
    # 
    # For spot i, we can take this weighted sum via:
    #
    #       np.einsum('i,ijk', K[i], O)
    # 
    # To do this over all spots, we broadcast it via:
    #
    #       np.einsum('...i,ijk', K, O)
    #
    all_covs = np.einsum('...i,ijk', K, O)
    all_covs = (all_covs.T / np.sum(K, axis=1)).T
    return all_covs


def _compute_kernel_matrix(
        df, 
        sigma, 
        cell_type_key='cluster', 
        condition_on_cell_type=False,
        y_col='imagerow',
        x_col='imagecol'
    ):
    # Get pixel coordinates
    coords = np.array(df[[y_col, x_col]])

    # Euclidean distance matrix
    dist_matrix = euclidean_distances(coords)

    # Compute matrix conditioning on cell type
    if not condition_on_cell_type:
        eta = np.full(dist_matrix.shape, 1)
    else:
        eta = []
        for ct1 in df[cell_type_key]:
            r = []
            for ct2 in df[cell_type_key]:
                r.append(int(ct1 == ct2)) 
            eta.append(r)
        eta = np.array(eta)

    # Gaussian kernel matrix
    kernel_matrix = np.exp(-1 * np.power(dist_matrix,2) / sigma**2)
    kernel_matrix = np.multiply(kernel_matrix, eta)

    return kernel_matrix


def _permute_expression(expr_1, expr_2, n_perms):
    expr = np.array([expr_1, expr_2]).T
    perms = np.array([
        np.random.permutation(expr)
        for i in range(n_perms)
    ])
    return perms


def _permute_expression_cond_cell_type(X, ct_to_indices, n_perms):
    """
    parameters
    ----------
    X
        GxN matrix of expression values where G is the number of genes
        and N is the number of spots.
    """
    expr = X.T
    perms = np.zeros((n_perms, len(expr), len(X.T[0])))
    for ct, indices in ct_to_indices.items():
        ct_expr = expr[indices]
        ct_perms = np.array([
            np.random.permutation(ct_expr)
            for i in range(n_perms)
        ])
        perms[:,indices,:] = ct_perms
    return perms

def _chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns
#import log_likelihood_ratio_test_plot as llrtp
def plot_mvn(mean, covar, X, the_x, title=''):
    color_iter = llrtp.PALETTE
    fig, ax = plt.subplots(1,1, figsize=(5,5))

    v, w = np.linalg.eigh(covar)
    v = 2. * np.sqrt(2.) * np.sqrt(v)
    u = w[0] / np.linalg.norm(w[0])

    ax.scatter(X[:, 0], X[:, 1], .8, color='black')
    ax.scatter([the_x[0]], [the_x[1]], 5., color='red')
    # Plot an ellipse to show the Gaussian component
    angle = np.arctan(u[1] / u[0])
    angle = 180. * angle / np.pi  # convert to degrees
    ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color='blue')
    ell.set_clip_box(ax.bbox)
    ell.set_alpha(0.2)
    ax.add_artist(ell)
    plt.show()


def _worker_between(
        worker_id,
        global_t_nulls,
        spotwise_t_nulls,
        df_filt,
        perms,
        kernel_matrix,
        ct_to_indices,
        null_corrs_filt,
        keep_indices,
        verbose=10,
        compute_spotwise_pvals=False
    ):
    """
    This function computes the test statistic on a chunk of permutations when
    run in parallel mode.
    """
    if verbose > 1:
        print(f"Started worker {worker_id}...")
    for perm_i, perm in enumerate(perms):
        if verbose > 1 and perm_i % 25 == 0:
            print(f"Worker {worker_id}, running permutation {perm_i}/{len(perms)}")
        # Compute alternative likelihoods
        perm_ll, perm_spot_lls  = compute_llrts_between(
            df_filt,
            perm.T,
            kernel_matrix,
            ct_to_indices,
            #alt_corrs_filt,
            null_corrs_filt,
            keep_indices
        )
        # Record the test statistic for this null sample
        global_t_nulls.append(perm_ll)
        if compute_spotwise_pvals:
            spotwise_t_nulls.append(perm_spot_lls)


def _worker_within(
        worker_id,
        global_t_nulls,
        spotwise_t_nulls,
        df_filt,
        perms,
        kernel_matrix,
        null_corrs_filt,
        keep_indices,
        verbose=10,
        compute_spotwise_pvals=False
    ):
    """
    This function computes the test statistic on a chunk of permutations when
    run in parallel mode.
    """
    if verbose > 1:
        print(f"Started worker {worker_id}...")
    for perm_i, perm in enumerate(perms):
        if verbose > 1 and perm_i % 25 == 0:
            print(f"Worker {worker_id}, running permutation {perm_i}/{len(perms)}")
        # Compute alternative likelihoods
        perm_ll, perm_spot_lls  = compute_llrts_within(
            df_filt,
            perm.T,
            kernel_matrix,
            null_corrs_filt,
            keep_indices
        )
        # Record the test statistic for this null sample
        global_t_nulls.append(perm_ll)
        if compute_spotwise_pvals:
            spotwise_t_nulls.append(perm_spot_lls)


def _adjust_covs_from_corrs(covs, corrs):
    """
    Given a list of covariance matrices and a list of 
    correlation matrices, for each pair, compute the 
    new covariances based on the based on the diagonal 
    (i.e. variance) in  to match the corresponding
    entries in the correlation matrix.

    Parameters
    ----------
    corrs
        A sequence of Pearson correlation matrices
    covs
        A sequence of covariance matrices
    
    Returns
    -------
    A sequence of adjusted covariance matrices
    """
    new_covs = []
    for cov, corr in zip(covs, corrs):
        varss = np.diag(cov)
        var_prods = np.sqrt(np.outer(varss, varss))
        new_cov = corr * var_prods
        np.fill_diagonal(new_cov, varss)
        new_covs.append(new_cov)
    new_covs = np.array(new_covs)
    return new_covs


def compute_llrts_between(
        df_filt,
        expr,
        kernel_matrix,
        ct_to_indices,
        #alt_corrs_filt,
        null_corrs_filt,
        keep_indices
    ):
    # Map index to cell type
    index_to_ct = {}
    for ct, indices in ct_to_indices.items():
        for index in indices:
            index_to_ct[index] = ct

    # Compute the local means
    means = (np.matmul(kernel_matrix, expr.T).T / np.sum(kernel_matrix, axis=1)).T
    means_filt = means[keep_indices]

    # Estimate covariance matrices using dynamic covariance
    dyn_covs = kernel_estimation(
        kernel_matrix,
        expr,
        verbose=False
    )
    dyn_covs_filt = dyn_covs[keep_indices]

    # Compute Pearson correlation matrix for spots within each 
    # cell type under the alternative hypothesis that there is differential
    # correlation between groups/cell types
    ct_to_corr = {
        ct: np.corrcoef(expr[:,indices])
        for ct, indices in ct_to_indices.items()
    }
    
    # Check each cluster to see if its covariance matrix is singular
    for ct, corr in ct_to_corr.items():
        if np.isnan(np.sum(corr)):
            print(f"Cluster '{ct}' has a singular covariance matrix. Removing spots from this cluster")
            keep_indices = sorted(set(keep_indices) - set(ct_to_indices[ct]))

    # Create the alternative correlation matrix at each spot. All spots
    # of the same cell type share the same correlation matrix
    alt_ct_corrs = np.array([
        ct_to_corr[index_to_ct[ind]]
        for ind in np.arange(len(expr.T))
    ])

    # Restrict data structures to only the kept indices
    alt_corrs_filt = alt_ct_corrs[keep_indices]
    
    # Compute alternative covariance matrices from correlations
    # and variances
    alt_covs_filt = _adjust_covs_from_corrs(dyn_covs_filt, alt_corrs_filt)

    # Build the inverse covariance matrices
    inv_alt_cov_mats_filt = np.array([
        np.linalg.inv(cov)
        for cov in alt_covs_filt
    ])

    # Build the log-likelihood function for each spot
    ll_alt_pdfs_filt = [
        build_normal_log_pdf(mean, cov, inv_cov)
        for bc, mean, cov, inv_cov in zip(
            df_filt.index,
            means_filt,
            alt_covs_filt,
            inv_alt_cov_mats_filt
        )
    ]

    # Compute the null covariance matrices.
    null_covs_filt = _adjust_covs_from_corrs(dyn_covs_filt, null_corrs_filt)

    # Build the inverse covariance matrices
    inv_null_cov_mats_filt = np.array([
        np.linalg.inv(cov)
        for cov in null_covs_filt
    ])

    # Build the log-likelihood function for each spot
    ll_null_pdfs_filt = [
        build_normal_log_pdf(mean, cov, inv_cov)
        for bc, mean, cov, inv_cov in zip(
            df_filt.index,
            means_filt,
            null_covs_filt,
            inv_null_cov_mats_filt
        )
    ]

    spot_null_lls_filt = np.array([
        ll(e)
        for ll, e in zip(ll_null_pdfs_filt, expr.T[keep_indices])
    ])
    spot_alt_lls_filt = np.array([
        ll(e)
        for ll, e in zip(ll_alt_pdfs_filt, expr.T[keep_indices])
    ])
    return np.sum(spot_alt_lls_filt - spot_null_lls_filt), list(spot_alt_lls_filt - spot_null_lls_filt)


def compute_llrts_within(
        df_filt,
        expr,
        kernel_matrix,
        null_corrs_filt,
        keep_indices,
        plot_ind=None
    ):
    """
    Parameters
    ----------
    df_filt
        Dataframe with metadata for spots that we are keeping.
    expr
        GxN matrix of expression where N is the total number of
        spots, not just the kept spots.
    kernel_matrix
        NxN matrix of pair-wise weights between spots
    null_corrs_filt
        
    """
    # Compute the local means
    means = (np.matmul(kernel_matrix, expr.T).T / np.sum(kernel_matrix, axis=1)).T
    means_filt = means[keep_indices]

    # Estimate covariance matrices using dynamic covariance
    dyn_covs = kernel_estimation(
        kernel_matrix,
        expr,
        verbose=False
    )
    dyn_covs_filt = dyn_covs[keep_indices]

    # Compute the null covariance matrix.
    # 'corr' is a Pearson correlation matrix 
    # and 'cov' is a covariance matrix. We want to 
    # compute the new covariances based on the 
    # diagonal in 'cov' to match corr.
    null_covs_filt = []
    for cov, corr in zip(dyn_covs_filt, null_corrs_filt):
        varss = np.diag(cov)
        var_prods = np.sqrt(np.outer(varss, varss))
        new_cov = corr * var_prods
        np.fill_diagonal(new_cov, varss)
        null_covs_filt.append(new_cov)
    null_covs_filt = np.array(null_covs_filt)
  
    # Build the inverse covariance matrices
    inv_null_cov_mats_filt = np.array([
        np.linalg.inv(cov)
        for cov in null_covs_filt
    ])

    # Build the log-likelihood function for each spot
    ll_null_pdfs_filt = [
        build_normal_log_pdf(mean, cov, inv_cov)
        for bc, mean, cov, inv_cov in zip(
            df_filt.index,
            means_filt,
            null_covs_filt,
            inv_null_cov_mats_filt
        )
    ]
  
    ll_alt_pdfs_filt = []
    for s_i, (bc, mean, cov) in enumerate(zip(df_filt.index, means_filt, dyn_covs_filt)):
        try:
            pdf = build_normal_log_pdf(mean, cov, np.linalg.inv(cov))
        except np.linalg.LinAlgError:
            print(f"Singular covariance matrix at spot {s_i}. Falling back on null covariance.")
            pdf = ll_null_pdfs_filt[s_i]
        ll_alt_pdfs_filt.append(pdf)

    spot_null_lls_filt = np.array([
        ll(e)
        for ll, e in zip(ll_null_pdfs_filt, expr.T[keep_indices])
    ])
    spot_alt_lls_filt = np.array([
        ll(e)
        for ll, e in zip(ll_alt_pdfs_filt, expr.T[keep_indices])
    ])
    return np.sum(spot_alt_lls_filt - spot_null_lls_filt), list(spot_alt_lls_filt - spot_null_lls_filt)


def _between_groups_test(
        expr,
        df,
        kernel_matrix,
        ct_to_indices,
        verbose=10,
        n_procs=1,
        keep_indices=None,
        use_sequential=True, 
        sequential_n_greater=20, 
        sequential_bail_out=10000,
        compute_spotwise_pvals=False,
        standardize_var=False,
        mc_pvals=True,
        spot_to_neighbors=None
    ):
    if keep_indices is None:
        keep_indices = np.arange(kernel_matrix.shape[0])

    # Subtract the means of each cell type
    ct_to_means = {
        ct: np.mean(expr[:,indices], axis=1)
        for ct, indices in ct_to_indices.items()
    }
    for ct, indices in ct_to_indices.items():
        expr[:,indices] = (expr[:,indices].T - np.full(expr[:,indices].T.shape, ct_to_means[ct])).T

    if standardize_var:
        ct_to_std = {
            ct: np.std(expr[:,indices], axis=1)
            for ct, indices in ct_to_indices.items()
        }
        for ct, indices in ct_to_indices.items():
            expr[:,indices] = (expr[:,indices].T / np.full(expr[:,indices].T.shape, ct_to_std[ct])).T

    # Compute the null correlation matrix for each spot. This is
    # a constant correlation matrix
    null_corr = np.corrcoef(expr)
    #print("Null corr: ", null_corr)
    null_corrs = np.array([
        null_corr
        for ind in np.arange(len(expr.T))
    ])

    # Restrict data structures to only the kept indices
    df_filt = df.iloc[keep_indices]
    #alt_corrs_filt = alt_ct_corrs[keep_indices]
    null_corrs_filt = null_corrs[keep_indices]

    obs_ll, obs_spot_lls = compute_llrts_between(
        df_filt,
        expr,
        kernel_matrix,
        ct_to_indices,
        #alt_corrs_filt,
        null_corrs_filt,
        keep_indices
    )
 
    # Observed statistic
    t_obs = obs_ll

    stop_monte_carlo = False # Whether to stop the sequential Monte Carlo calculations
    n_nulls_great = 0        # Number of null statistics greater than the observed
    PERM_SIZE = 100           # Every batch, we compute a set number of permutations
    all_t_nulls = []
    all_spotwise_t_nulls = []
    while not stop_monte_carlo:
        # Compute permutations conditioned on cell type
        perms = _permute_expression_cond_cell_type(
            expr,
            {'all': list(np.arange(len(expr.T)))}, # Permute all of the spots
            PERM_SIZE
        )

        # Sample statistic from null distribution
        if n_procs > 1: # Multi-processing

            manager = Manager()
            t_nulls = manager.list()
            spotwise_t_nulls = manager.list()

            chunk_size = math.ceil(len(perms) / n_procs)
            if verbose > 1:
                print(f"Chunk size is {chunk_size}")
            perm_chunks = _chunks(perms, chunk_size)
            jobs = []
            for worker_id, chunk in enumerate(perm_chunks):
                p = Process(
                    target=_worker_between,
                    args=(
                        worker_id,
                        t_nulls,
                        spotwise_t_nulls,
                        df_filt,
                        chunk,
                        kernel_matrix,
                        ct_to_indices,
                        #alt_corrs_filt,
                        null_corrs_filt,
                        keep_indices,
                        verbose,
                        compute_spotwise_pvals
                    )
                )
                jobs.append(p)
                p.start()
            for p_i, p in enumerate(jobs):
                if verbose > 2:
                    print(f"Worker {p_i} finished.")
                p.join()


        else:
            t_nulls = []
            spotwise_t_nulls = []
            for perm_i, perm in enumerate(perms):
                if verbose > 1 and perm_i % 10 == 0:
                    print('Computing statistic for permutation {}/{}'.format(perm_i+1, len(perms)))

                # Compute alternative likelihoods
                perm_ll, perm_spot_lls  = compute_llrts_between(
                    df_filt,
                    perm.T,
                    kernel_matrix,
                    ct_to_indices,
                    #alt_corrs_filt,
                    null_corrs_filt,
                    keep_indices
                )

                # Record the test statistic for this null sample
                t_nulls.append(perm_ll)
                if compute_spotwise_pvals:
                    spotwise_t_nulls.append(perm_spot_lls)

        # Add all new spotwise statistics for each permutation
        # to the full collection
        all_spotwise_t_nulls += spotwise_t_nulls

        if mc_pvals:
            for t_null in t_nulls:
                all_t_nulls.append(t_null)
                if t_null > t_obs:
                    n_nulls_great += 1
                if n_nulls_great == sequential_n_greater:
                    hit_threshold = True
                    p_val = n_nulls_great / len(all_t_nulls)
                    print(f"Number of nulls > obs has hit threshold of {sequential_n_greater}. Total permutations used: {len(all_t_nulls)}. P-value = {p_val}")
                    stop_monte_carlo = True
                    break
                if len(all_t_nulls) >= (sequential_bail_out-1):
                    p_val = (n_nulls_great + 1) / sequential_bail_out
                    print(f"Hit maximum permutations threshold of {sequential_n_greater}. P-value = {p_val}")
                    stop_monte_carlo = True
                    break
        else:
            for t_null in t_nulls:
                all_t_nulls.append(t_null)
                if t_null > t_obs:
                    n_nulls_great += 1
                if len(all_t_nulls) >= (sequential_bail_out-1):
                    p_val = (n_nulls_great + 1) / sequential_bail_out
                    stop_monte_carlo = True
                    break

    spot_p_vals = None
    if compute_spotwise_pvals:
        # Create an NxP array where N is number of spots
        # P is number of null samples where each row stores
        # the null spotwise statistics for each spot.
        # Because we may have more permutations than we used (due
        # to computation of Monte-Carlo p-values), we restrict to 
        # only number of permutations used.
        all_spotwise_t_nulls = np.array(all_spotwise_t_nulls)[:len(all_t_nulls),:]
        all_spotwise_t_nulls = all_spotwise_t_nulls.T
        spot_neigh_t_nulls = []
        obs_neight_lls = []
        if spot_to_neighbors:
            spot_to_index = {
                spot: s_i
                for s_i, spot in enumerate(df_filt.index)
            }
            for spot in df_filt.index:
                neighs = set(spot_to_neighbors[spot]) | set([spot])
                neigh_inds = [spot_to_index[x] for x in neighs if x in spot_to_index]
                obs_neight_lls.append(np.sum(np.array(obs_spot_lls)[neigh_inds]))
                spot_neigh_t_nulls.append(np.sum(all_spotwise_t_nulls[neigh_inds,:], axis=0))
            spot_neigh_t_nulls = np.array(spot_neigh_t_nulls)
            assert spot_neigh_t_nulls.shape == all_spotwise_t_nulls.shape

            # Compute spot-wise p-values using neighborhood-summed log-likelihoods at each spot
            spot_p_vals = []
            for obs_spot_ll, null_spot_lls in zip(obs_neight_lls, spot_neigh_t_nulls):
                # The +1 in the numerator and denominator is the observed test statistic
                spot_p_val = (len([x for x in null_spot_lls if x > obs_spot_ll])+1) / (len(null_spot_lls)+1)
                spot_p_vals.append(spot_p_val)
        else:
            spot_p_vals = []
            for obs_spot_ll, null_spot_lls in zip(obs_spot_lls, all_spotwise_t_nulls):
                spot_p_val = (len([x for x in null_spot_lls if x > obs_spot_ll])+1) / (len(null_spot_lls)+1)
                spot_p_vals.append(spot_p_val)
    return p_val, t_obs, t_nulls, obs_spot_lls, all_spotwise_t_nulls, spot_p_vals




def _within_groups_test(
        expr, 
        df, 
        kernel_matrix, 
        plot_lls=True, 
        verbose=False,
        n_procs=1,
        ct_to_indices=None,
        ct_to_indices_filt=None,
        keep_indices=None,
        use_sequential=True, 
        sequential_n_greater=20, 
        sequential_bail_out=10000,
        compute_clust_pvals=False,
        mc_pvals=True,
        spot_to_neighbors=None
    ):
    """
    t_nulls, t_obs, p_val, spot_obs_ll_diffs, spot_perm_ll_diffs

    Note that the ll_diffs are ll_alt - ll_null, so the higher
    ll_diff, the more likely to come form the alternative.
    """

    if keep_indices is None:
        keep_indices = np.arange(kernel_matrix.shape[0])

    # Create a "cell type" consisting of all spots
    if ct_to_indices is None:
        ct_to_indices = {'all': list(np.arange(len(expr.T)))}    

    # Map index to cell type
    index_to_ct = {}
    for ct, indices in ct_to_indices.items():
        for index in indices:
            index_to_ct[index] = ct

    # Compute Pearson correlation matrix for spots within each 
    # cell type under the null hypothesis that there is no spatial
    # correlation
    ct_to_corr = {
        ct: np.corrcoef(expr[:,indices])
        for ct, indices in ct_to_indices.items()
    }

    # Check each cluster to see if its covariance matrix is singular
    for ct, corr in ct_to_corr.items():
        if np.isnan(np.sum(corr)):
            print(f"Cluster '{ct}' has a singular covariance matrix. Removing spots from this cluster")
            keep_indices = sorted(set(keep_indices) - set(ct_to_indices[ct]))

    # Create the null correlation matrix at each spot
    null_ct_corrs = np.array([
        ct_to_corr[index_to_ct[ind]]
        for ind in np.arange(len(expr.T))
    ])

    # Restrict data structures to only the kept indices
    df_filt = df.iloc[keep_indices]
    null_corrs_filt = null_ct_corrs[keep_indices]

    # Compute the log-likelihood ratios for the observed data
    obs_ll, obs_spot_lls = compute_llrts_within(
        df_filt,
        expr,
        kernel_matrix,
        null_corrs_filt,
        keep_indices,
        plot_ind=67
    )
  
    # Observed statistic
    t_obs = obs_ll

    stop_monte_carlo = False # Whether to stop the sequential Monte Carlo calculations
    n_nulls_great = 0        # Number of null statistics greater than the observed
    PERM_SIZE = 100          # Every batch, we compute a set number of permutations
    all_t_nulls = []
    all_spotwise_t_nulls = []
    while not stop_monte_carlo:
        # Compute permutations conditioned on cell type
        perms = _permute_expression_cond_cell_type(
            expr,
            ct_to_indices,
            PERM_SIZE
        )

        # Sample statistic from null distribution
        if n_procs > 1: # Multi-processing

            manager = Manager()
            t_nulls = manager.list()
            spotwise_t_nulls = manager.list()

            chunk_size = math.ceil(len(perms) / n_procs)
            if verbose > 1:
                print(f"Chunk size is {chunk_size}")
            perm_chunks = _chunks(perms, chunk_size)
            jobs = []
            for worker_id, chunk in enumerate(perm_chunks):
                p = Process(
                    target=_worker_within,
                    args=(
                        worker_id,
                        t_nulls,
                        spotwise_t_nulls,
                        df_filt,
                        chunk,
                        kernel_matrix,
                        null_corrs_filt,
                        keep_indices,
                        verbose,
                        compute_clust_pvals
                    )
                )
                jobs.append(p)
                p.start()
            for p_i, p in enumerate(jobs):
                if verbose > 2:
                    print(f"Worker {p_i} finished.")
                p.join()
        else:
            t_nulls = []
            spotwise_t_nulls = []
            for perm_i, perm in enumerate(perms):
                if verbose > 1 and perm_i % 10 == 0:
                    print('Computing ratio statistic for permutation {}/{}'.format(perm_i+1, len(perms)))

                # Compute alternative likelihoods
                perm_ll, perm_spot_lls  = compute_llrts_within(
                    df_filt,
                    perm.T,
                    kernel_matrix,
                    null_corrs_filt,
                    keep_indices
                )

                # Record the test statistic for this null sample
                t_nulls.append(perm_ll)
                if compute_clust_pvals:
                    spotwise_t_nulls.append(perm_spot_lls)

        # Add all new spotwise statistics for each permutation
        # to the full collection
        all_spotwise_t_nulls += spotwise_t_nulls

        if mc_pvals:
            for t_null in t_nulls:
                all_t_nulls.append(t_null)
                if t_null > t_obs:
                    n_nulls_great += 1
                if n_nulls_great == sequential_n_greater:
                    hit_threshold = True
                    p_val = n_nulls_great / len(all_t_nulls)
                    if verbose > 0:
                        print(f"Number of nulls > obs has hit threshold of {sequential_n_greater}. Total permutations used: {len(all_t_nulls)}. P-value = {p_val}")
                    stop_monte_carlo = True
                    break
                if len(all_t_nulls) >= (sequential_bail_out-1):
                    p_val = (n_nulls_great + 1) / sequential_bail_out
                    if verbose > 0:
                        print(f"Hit maximum permutations threshold of {sequential_n_greater}. P-value = {p_val}")
                    stop_monte_carlo = True
                    break
        else:
            for t_null in t_nulls:
                all_t_nulls.append(t_null)
                if t_null > t_obs:
                    n_nulls_great += 1
                if len(all_t_nulls) >= (sequential_bail_out-1):
                    p_val = (n_nulls_great + 1) / sequential_bail_out
                    stop_monte_carlo = True
                    break

    ct_to_p_val = None
    if compute_clust_pvals:
        # Create an NxP array where N is number of spots
        # P is number of null samples where each row stores
        # the null spotwise statistics for each spot.
        # Because we may have more permutations than we used (due
        # to computation of Monte-Carlo p-values), we restrict to
        # only number of permutations used.
        all_spotwise_t_nulls = np.array(all_spotwise_t_nulls)[:len(all_t_nulls),:]
        all_spotwise_t_nulls = all_spotwise_t_nulls.T

        ct_to_obs = {
            ct: np.sum(np.array(obs_spot_lls)[inds])
            for ct, inds in ct_to_indices_filt.items()
        }
        ct_to_nulls = {
            ct: np.sum(all_spotwise_t_nulls[inds,:], axis=0)
            for ct, inds in ct_to_indices_filt.items()
        }
        ct_to_p_val = {
            ct: (len([x for x in ct_to_nulls[ct] if x > ct_to_obs[ct]])+1) / (len(ct_to_nulls[ct])+1)
            for ct in ct_to_nulls
        }
    

    #spot_p_vals = None
    #if compute_spotwise_pvals:
    #    # Create an NxP array where N is number of spots
    #    # P is number of null samples where each row stores
    #    # the null spotwise statistics for each spot.
    #    # Because we may have more permutations than we used (due
    #    # to computation of Monte-Carlo p-values), we restrict to 
    #    # only number of permutations used.
    #    all_spotwise_t_nulls = np.array(all_spotwise_t_nulls)[:len(all_t_nulls),:]
    #    all_spotwise_t_nulls = all_spotwise_t_nulls.T
    #    spot_neigh_t_nulls = []
    #    obs_neight_lls = []
    #    if spot_to_neighbors:
    #        spot_to_index = {
    #            spot: s_i
    #            for s_i, spot in enumerate(df_filt.index)
    #        }
    #        for spot in df_filt.index:
    #            neighs = set(spot_to_neighbors[spot]) | set([spot])
    #            neigh_inds = [spot_to_index[x] for x in neighs if x in spot_to_index]
    #            obs_neight_lls.append(np.sum(np.array(obs_spot_lls)[neigh_inds]))
    #            spot_neigh_t_nulls.append(np.sum(all_spotwise_t_nulls[neigh_inds,:], axis=0))
    #        spot_neigh_t_nulls = np.array(spot_neigh_t_nulls)
    #        assert spot_neigh_t_nulls.shape == all_spotwise_t_nulls.shape
    #
    #        # Compute spot-wise p-values using neighborhood-summed log-likelihoods at each spot
    #        spot_p_vals = []
    #        for obs_spot_ll, null_spot_lls in zip(obs_neight_lls, spot_neigh_t_nulls):
    #            spot_p_val = (len([x for x in null_spot_lls if x > obs_spot_ll])+1) / (len(null_spot_lls)+1)
    #            spot_p_vals.append(spot_p_val)
    #    else:
    #        spot_p_vals = []
    #        for obs_spot_ll, null_spot_lls in zip(obs_spot_lls, all_spotwise_t_nulls):
    #            spot_p_val = (len([x for x in null_spot_lls if x > obs_spot_ll])+1) / (len(null_spot_lls)+1) 
    #            spot_p_vals.append(spot_p_val)
    return p_val, t_obs, t_nulls, obs_spot_lls, all_spotwise_t_nulls, ct_to_p_val


def run_test(
        adata,
        test_genes,
        sigma,
        cond_key=None,
        contrib_thresh=10,
        row_key='row',
        col_key='col',
        verbose=1,
        n_procs=1,
        test_between_conds=False,
        compute_spotwise_pvals=False,
        standardize_var=False,
        max_perms=10000,
        mc_pvals=True,
        spot_to_neighbors=None
    ):

    # Extract expression data
    expr = np.array([
        adata.obs_vector(gene)
        for gene in test_genes
    ])

    condition = cond_key is not None

    # Compute kernel matrix
    kernel_matrix = _compute_kernel_matrix(
        adata.obs,
        sigma=sigma,
        cell_type_key=cond_key,
        condition_on_cell_type=condition,
        y_col=row_key,
        x_col=col_key
    )

    # Filter spots with too little contribution 
    # from neighbors
    contrib = np.sum(kernel_matrix, axis=1)
    keep_inds = [
        i
        for i, c in enumerate(contrib)
        if c >= contrib_thresh
    ]
    if verbose >= 1:
        print('Kept {}/{} spots.'.format(len(keep_inds), len(adata.obs)))

    # Map each cell type to its indices in the full dataset
    if condition:
        ct_to_indices = defaultdict(lambda: [])
        for i, ct in enumerate(adata.obs[cond_key]):
            ct_to_indices[ct].append(i)
    else:
        ct_to_indices = {'all': np.arange(len(adata.obs))}

    # Mape each cell type to the its indices in the filtered
    # dataset
    ct_to_indices_filt = defaultdict(lambda: [])
    for filt_ind, ct in enumerate(adata.obs.iloc[keep_inds][cond_key]):
        ct_to_indices_filt[ct].append(filt_ind)

    if test_between_conds:
        assert condition
        p_val, t_obs, t_nulls, obs_spot_lls, spotwise_t_nulls, spot_p_vals = _between_groups_test(
            expr,
            adata.obs,
            kernel_matrix,
            ct_to_indices,
            verbose=verbose,
            n_procs=n_procs,
            keep_indices=keep_inds,
            compute_spotwise_pvals=compute_spotwise_pvals,
            sequential_bail_out=max_perms,
            standardize_var=standardize_var,
            mc_pvals=mc_pvals,
            spot_to_neighbors=spot_to_neighbors
        )
    else:
        p_val, t_obs, t_nulls, obs_spot_lls, spotwise_t_nulls, spot_p_vals = _within_groups_test(
            expr,
            adata.obs,
            kernel_matrix,
            plot_lls=False,
            ct_to_indices=ct_to_indices,
            ct_to_indices_filt=ct_to_indices_filt,
            verbose=verbose,
            n_procs=n_procs,
            keep_indices=keep_inds,
            compute_clust_pvals=compute_spotwise_pvals, # TODO change variable name
            sequential_bail_out=max_perms,
            mc_pvals=mc_pvals,
            spot_to_neighbors=spot_to_neighbors
        )
    return p_val, t_obs, t_nulls, keep_inds, obs_spot_lls, spotwise_t_nulls, spot_p_vals


def run_test_between_region_pairs(
        adata,
        test_genes,
        sigma,
        cond_key,
        contrib_thresh=10,
        row_key='row',
        col_key='col',
        verbose=1,
        n_procs=1,
        standardize_var=False,
        max_perms=10000,
        mc_pvals=True,
        spot_to_neighbors=None,
        run_regions=None
    ):

    # Filter spots with too little contribution 
    # from neighbors
    # Compute kernel matrix
    kernel_matrix = _compute_kernel_matrix(
        adata.obs,
        sigma=sigma,
        cell_type_key=cond_key,
        condition_on_cell_type=True,
        y_col=row_key,
        x_col=col_key
    )
    contrib = np.sum(kernel_matrix, axis=1)
    keep_inds = [
        i
        for i, c in enumerate(contrib)
        if c >= contrib_thresh
    ]
    if verbose >= 1:
        print('Kept {}/{} spots.'.format(len(keep_inds), len(adata.obs)))
    adata = adata[keep_inds,:]

    # Map each cell type to its indices
    ct_to_indices = defaultdict(lambda: [])
    for i, ct in enumerate(adata.obs[cond_key]):
        ct_to_indices[ct].append(i)

    # If the regionst to run aren't specified, run on all pairs
    if run_regions is None:
        run_regions = sorted(set(adata.obs[cond_key]))

    ct_to_ct_to_pval = defaultdict(lambda: {})
    for ct_1_i, ct_1 in enumerate(sorted(run_regions)):
        for ct_2_i, ct_2 in enumerate(sorted(run_regions)):
            if ct_2 >= ct_1:
                continue

            clust_inds = list(ct_to_indices[ct_1]) + list(ct_to_indices[ct_2])
            adata_clust = adata[clust_inds,:]

            # Extract expression data
            expr = np.array([
                adata_clust.obs_vector(gene)
                for gene in test_genes
            ]) 

            # Compute kernel matrix
            kernel_matrix_clust = _compute_kernel_matrix(
                adata_clust.obs,
                sigma=sigma,
                cell_type_key=cond_key,
                condition_on_cell_type=True,
                y_col=row_key,
                x_col=col_key
            )

            # Filter spots with too little contribution 
            # from neighbors
            contrib = np.sum(kernel_matrix_clust, axis=1)
            keep_inds = [
                i
                for i, c in enumerate(contrib)
                if c >= contrib_thresh
            ]
            if verbose >= 1:
                print('For cluster pair ({}, {}), kept {}/{} spots.'.format(
                    ct_1,
                    ct_2,
                    len(keep_inds), 
                    len(adata_clust.obs)
                ))    

            # Re-map each cluster to its indices now that we have restricted
            # the dataset to only two clusters
            ct_to_indices_clust = defaultdict(lambda: [])
            for i, ct in enumerate(adata_clust.obs[cond_key]):
                ct_to_indices_clust[ct].append(i)

            p_val, t_obs, t_nulls, obs_spot_lls, spotwise_t_nulls, spot_p_vals = _between_groups_test(
                expr,
                adata_clust.obs,
                kernel_matrix_clust,
                ct_to_indices_clust, 
                verbose=verbose,
                n_procs=n_procs,
                keep_indices=keep_inds,
                compute_spotwise_pvals=False,
                sequential_bail_out=max_perms,
                standardize_var=standardize_var,
                mc_pvals=mc_pvals,
                spot_to_neighbors=None
            )

            ct_to_ct_to_pval[ct_1][ct_2] = p_val
            ct_to_ct_to_pval[ct_2][ct_1] = p_val
    return ct_to_ct_to_pval
   

def est_corr_cis(g1, g2, ad, kernel_matrix, cond_key, neigh_thresh=10, bc_to_neighs=None, n_boots=100, mag=0):
    expr_1 = ad.obs_vector(g1)
    expr_2 = ad.obs_vector(g2)
    if bc_to_neighs is None:
        row_col_to_barcode = spatialcorr.utils.map_row_col_to_barcode(
            ad.obs,
            row_key='row',
            col_key='col'
        )
        bc_to_neighs = spatialcorr.utils.compute_neighbors(
            ad.obs,
            row_col_to_barcode,
            row_key='row',
            col_key='col',
            rad=3
        )
    
    clust_to_bcs = defaultdict(lambda: set())
    for bc, clust in zip(ad.obs.index, ad.obs[cond_key]):
        clust_to_bcs[clust].add(bc)
                             
    bc_to_clust = {
        bc: clust
        for bc, clust in zip(ad.obs.index, ad.obs[cond_key])
    }
    keep_inds = []
    bin_corrs = []
    bc_to_ind = {
        bc: bc_i
        for bc_i, bc in enumerate(ad.obs.index)
    }
    cis = []
    for bc_i, bc in enumerate(ad.obs.index):
        neighs = bc_to_neighs[bc]
        neighs = set(neighs) | set([bc]) # Add current spot
        curr_clust = bc_to_clust[bc]
        neighs = sorted(set(neighs) & clust_to_bcs[curr_clust])
        if len(neighs) < neigh_thresh:
            continue
        keep_inds.append(bc_i)
        neigh_inds = [bc_to_ind[x] for x in neighs]
        
        weights_neigh = kernel_matrix[bc_i][neigh_inds]
        expr_1_neigh = expr_1[neigh_inds]
        expr_2_neigh = expr_2[neigh_inds]
        
        boot_corrs = []
        for i in range(n_boots): 
            boot_inds = np.random.choice(
                np.arange(len(expr_1_neigh)), 
                size=len(expr_1_neigh), 
                replace=True
            )
            e_1_boot = expr_1_neigh[boot_inds]
            e_2_boot = expr_2_neigh[boot_inds]
            weights_boot = weights_neigh[boot_inds]
            
            mean_1_boot = sum(e_1_boot * weights_boot) / sum(weights_boot)
            mean_2_boot = sum(e_2_boot * weights_boot) / sum(weights_boot)
            var_1_boot = sum( np.power((e_1_boot - mean_1_boot), 2) * weights_boot) / sum(weights_boot)
            var_2_boot = sum( np.power((e_1_boot - mean_2_boot), 2) * weights_boot) / sum(weights_boot)
            cov_boot = sum( (e_1_boot - mean_1_boot) * (e_2_boot - mean_2_boot) * weights_boot) / sum(weights_boot)
            corr_boot = cov_boot / np.sqrt(var_1_boot * var_2_boot)
            
            boot_corrs.append(corr_boot)
        boot_corrs = sorted(boot_corrs)
        cis.append((boot_corrs[5], boot_corrs[95]))
        
    return cis, keep_inds
