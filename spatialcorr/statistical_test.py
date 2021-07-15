import matplotlib as mpl  
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
import math
import importlib
import seaborn as sns
from collections import defaultdict
from sklearn.metrics.pairwise import euclidean_distances
from multiprocessing import Process, Queue, Manager

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
        compute_spotwise_pvals=False
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
    PERM_SIZE = 50           # Every batch, we compute a set number of permutations
    all_t_nulls = []
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
            perm_chunks = chunks(perms, chunk_size)
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
                if verbose and perm_i % 10 == 0:
                    print('Computing ratio statistic for permutation {}/{}'.format(perm_i+1, len(perms)))

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
        # Create an NxP array where N is number of spots
        # P is number of null samples where each row stores
        # the null spotwise statistics for each spot
        spotwise_t_nulls = np.array(spotwise_t_nulls).T
        spot_p_vals = []
        for obs_spot_ll, null_spot_lls in zip(obs_spot_lls, spotwise_t_nulls):
            spot_p_val = len([x for x in null_spot_lls if x > obs_spot_ll]) / len(null_spot_lls)
            spot_p_vals.append(spot_p_val)

    return p_val, t_obs, t_nulls, obs_spot_lls, spotwise_t_nulls, spot_p_vals


def _within_groups_test(
        expr, 
        df, 
        kernel_matrix, 
        plot_lls=True, 
        verbose=False,
        n_procs=1,
        ct_to_indices=None,
        keep_indices=None,
        use_sequential=True, 
        sequential_n_greater=20, 
        sequential_bail_out=10000,
        compute_spotwise_pvals=False
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
    PERM_SIZE = 50           # Every batch, we compute a set number of permutations
    all_t_nulls = []
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
            perm_chunks = chunks(perms, chunk_size)
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
                if compute_spotwise_pvals:
                    spotwise_t_nulls.append(perm_spot_lls)

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

    # Create an NxP array where N is number of spots
    # P is number of null samples where each row stores
    # the null spotwise statistics for each spot
    spotwise_t_nulls = np.array(spotwise_t_nulls).T
    spot_p_vals = []
    for obs_spot_ll, null_spot_lls in zip(obs_spot_lls, spotwise_t_nulls):
        spot_p_val = len([x for x in null_spot_lls if x > obs_spot_ll]) / len(null_spot_lls) 
        spot_p_vals.append(spot_p_val)
    return p_val, t_obs, t_nulls, obs_spot_lls, spotwise_t_nulls, spot_p_vals


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
        max_perms=10000
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

    # Map each cell type to its indices
    if condition:
        ct_to_indices = defaultdict(lambda: [])
        for i, ct in enumerate(adata.obs[cond_key]):
            ct_to_indices[ct].append(i)
    else:
        ct_to_indices = {'all': np.arange(len(adata.obs))}

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
            sequential_bail_out=max_perms
        )
    else:
        p_val, t_obs, t_nulls, obs_spot_lls, spotwise_t_nulls, spot_p_vals = _within_groups_test(
            expr,
            adata.obs,
            kernel_matrix,
            plot_lls=False,
            ct_to_indices=ct_to_indices,
            verbose=verbose,
            n_procs=n_procs,
            keep_indices=keep_inds,
            compute_spotwise_pvals=compute_spotwise_pvals,
            sequential_bail_out=max_perms
        )
    return p_val, t_obs, t_nulls, keep_inds, obs_spot_lls, spotwise_t_nulls, spot_p_vals



#def log_likelihood_ratio_test(
#        expr,
#        df,
#        sigma,
#        cell_type_key='cluster',
#        cond_cell_type=False,
#        contrib_thresh=5,
#        x_col='col',
#        y_col='row',
#        verbose=10,
#        n_procs=1,
#        between_cell_types=False,
#        max_perms=10000
#    ):
#    # Compute kernel matrix
#    kernel_matrix = _compute_kernel_matrix(
#        df, 
#        sigma=sigma, 
#        cell_type_key=cell_type_key,
#        condition_on_cell_type=cond_cell_type,
#        y_col=y_col,
#        x_col=x_col
#    )
#
#    # Filter spots with too little contribution 
#    # from neighbors
#    contrib = np.sum(kernel_matrix, axis=1)
#    keep_inds = [
#        i
#        for i, c in enumerate(contrib)
#        if c >= contrib_thresh
#    ]
#    if verbose >= 1:
#        print('Kept {}/{} spots.'.format(len(keep_inds), len(df)))
#
#    # Map each cell type to its indices
#    if cond_cell_type:
#        ct_to_indices = defaultdict(lambda: [])
#        for i, ct in enumerate(df[cell_type_key]):
#            ct_to_indices[ct].append(i)
#    else:
#        ct_to_indices = {'all': np.arange(len(df))}
#
#    if between_cell_types:
#        assert cond_cell_type
#        p_val, t_obs, t_nulls, obs_spot_lls = _between_groups_test(
#            expr,
#            df,
#            kernel_matrix,
#            ct_to_indices,
#            verbose=verbose,
#            n_procs=n_procs,
#            keep_indices=keep_inds,
#            compute_spotwise_pvals=compute_spotwise_pvals,
#            sequential_bail_out=max_perms
#        )
#    else:
#        p_val, t_obs, t_nulls, obs_spot_lls = _within_groups_test(
#            expr,
#            df,
#            kernel_matrix,
#            plot_lls=False,
#            ct_to_indices=ct_to_indices,
#            verbose=verbose,
#            n_procs=n_procs,
#            keep_indices=keep_inds,
#            compute_spotwise_pvals=compute_spotwise_pvals,
#            sequential_bail_out=max_perms
#        )
#
#    return p_val, spot_p_vals, t_obs, t_nulls


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
