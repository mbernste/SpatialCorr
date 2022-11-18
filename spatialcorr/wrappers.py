"""
Wrapper functions implementing high-level pipelines/workflows for analyzing
and visualizing spatially varying correlation.

Authors: Matthew Bernstein <mbernstein@morgridge.org>
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from .statistical_test  import run_test, run_test_between_region_pairs, compute_kernel_matrix
from .plot import plot_slide, plot_correlation, plot_ci_overlap, plot_filtered_spots


def analysis_pipeline_pair(
        adata,
        gene_1, 
        gene_2,
        cond_key = 'cluster',
        bandwidth = 5,
        row_key='row',
        col_key='col',
        reject_thresh=0.05,
        dsize=12,
        max_perms=500, 
        n_procs=5, 
        contrib_thresh=10,
        verbose=1,
        fig_path=None,
        fig_format='pdf',
        dpi=150,
        cmap_expr='viridis',
        cmap_corr='RdBu_r',
        only_stats=False
    ):
    """Run a SpatialCorr analysis pipeline on a pair of genes.

    This function will run the following analyses:
    1. Compute spotwise kernel estimates of correlation
    2. Compute confidence intervals (CIs) of correlation at each spot compute spots where CI does not overlap zero (i.e. putative regions with non-zero correlation)
    3. For each cluster, compute a WR P-value
    4. Remove all clusters with WR P-value < `reject_thresh` for BR-test and for remaining clusters, compute BR P-value testing for differential correlation between the two clusters 

    Parameters
    ----------
    gene_1: string
        The first gene of the pair to analyze
    gene_2: string
        The second gene of the pair to analyze
    adata : AnnData
        spatial gene expression dataset with spatial coordinates
        stored in `adata.obs`
    bandwidth : int
        the kernel bandwidth used by the test
    cond_key : string
        the name of the column in `adata.obs` storing the cluster
        assignments
    row_key : string, optional (default: 'row')
        the name of the column in `adata.obs` storing the row coordinates
        of each spot
    col_key : string, optional (default: 'col')
        the name of the column in `adata.obs` storing the column
        coordinates of each spot
    reject_thresh: float (default: 0.05)
        P-value threshold used to reject the null hypothesis for each
        region's WR-test as well as region-pairwise BR-tests.
    dsize: int, optional (default: 12)
        the size of the dots in the scatterplot
    max_perms : int, optional (default: 500)
        Maximum number of permutations to compute for the permutation
        test
    n_procs : int, optional (default: 1)
        number of processes to run in parallel
    verbose : int, optional (default: 1)
        the verbosity. Higher verbosity will lead to more debugging
        information printed to standard output
    contrib_thresh : int, optional (default: 10)
        threshold for the  total weight of all samples contributing
        to the correlation estimate at each spot. Spots with total
        weight less than this value will be filtered prior to running
        the test
    fig_path: string, optional (default: None)
        Path to write figure image.
    fig_format: string, {'pdf', 'png'} (default: 'pdf')
        Format of the output figure file.
    dpi: int (default: 150)
        Resolution of output image.
    cmap_expr : String, optional (default 'turbo')
        colormap for expression figures.
    cmap_corr : String, optional (default 'RdBu_r')
        colormap for correlation figures.

    Returns
    -------
    None
    """

    if only_stats:
        fig, axarr = plt.subplots(
            2,
            2,
            figsize=(7,7)
        )
    else:
        fig, axarr = plt.subplots(
            2,
            4,
            figsize=(14,7)
        )  
    
    if not only_stats:
        expr_1 = adata.obs_vector(gene_1)
        expr_2 = adata.obs_vector(gene_2)
        min_expr = min(list(expr_1) + list(expr_2))
        max_expr = max(list(expr_1) + list(expr_2))
            
        plot_slide(
            adata.obs,
            expr_1,
            cmap=cmap_expr,
            colorbar=False,
            vmin=min_expr,
            vmax=max_expr,
            title=f'{gene_1} expression',
            ax=axarr[0][0],
            figure=None,
            ticks=False,
            dsize=dsize,
            colorticks=None,
            row_key=row_key,
            col_key=col_key
        )
        plot_slide(
            adata.obs,
            expr_2,
            cmap=cmap_expr,
            colorbar=False,
            vmin=min_expr,
            vmax=max_expr,
            title=f'{gene_2} expression',
            ax=axarr[1][0],
            figure=None,
            ticks=False,
            dsize=dsize,
            colorticks=None,
            row_key=row_key,
            col_key=col_key
        )
        plot_slide(
            adata.obs,
            adata.obs[cond_key],
            cmap='categorical',
            colorbar=True,
            vmin=min_expr,
            vmax=max_expr,
            title=f'Clusters',
            ax=axarr[0][1],
            figure=None,
            ticks=False,
            dsize=dsize,
            colorticks=None,
            row_key=row_key,
            col_key=col_key
        )

        # Plot filtered spots
        kernel_matrix = compute_kernel_matrix(
            adata.obs,
            bandwidth=bandwidth,
            y_col=row_key,
            x_col=col_key,
            condition_on_region=(not cond_key is None),
            region_key=cond_key
        ) 
        plot_filtered_spots(
            adata,
            kernel_matrix,
            contrib_thresh,
            row_key=row_key,
            col_key=col_key,
            ax=axarr[1][1],
            figure=None,
            dsize=dsize,
            ticks=False
        )
   
    if only_stats:
        ax = axarr[0][0]
    else:
        ax = axarr[0][2]
    plot_correlation(
        adata,
        gene_1,
        gene_2,
        bandwidth=bandwidth,
        contrib_thresh=contrib_thresh,
        row_key=row_key,
        col_key=col_key,
        condition=cond_key,
        cmap=cmap_corr,
        colorbar=False,
        ticks=False,
        ax=ax,
        figure=fig,
        dsize=dsize,
        estimate='local',
        title='Correlation\n(kernel estimate)'
    )
    
    if only_stats:
        ax = axarr[0][1]
    else:
        ax = axarr[0][3]
    plot_ci_overlap(
        adata,
        gene_1,
        gene_2,
        cond_key=cond_key,
        kernel_matrix=None,
        bandwidth=bandwidth,
        row_key=row_key,
        col_key=col_key,
        title='Regions of non-zero\ncorrelation',
        ax=ax,
        figure=None,
        ticks=False,
        dsize=dsize,
        colorticks=None,
        neigh_thresh=contrib_thresh
    )

    p_val, additional = run_test(
        adata,
        [gene_1, gene_2],
        bandwidth,
        cond_key=cond_key,
        contrib_thresh=contrib_thresh,
        row_key=row_key,
        col_key=col_key,
        verbose=verbose,
        n_procs=n_procs,
        run_br=False,
        compute_spotwise_pvals=True,
        max_perms=max_perms,
        mc_pvals=False
    )
    kept_inds = additional['kept_inds']
    clust_to_p_val = additional['region_to_p_val']
    
    spot_p_vals = [
        clust_to_p_val[ct]
        for ct in adata.obs.iloc[kept_inds][cond_key]
    ]

    if only_stats:
        ax = axarr[1][0]
    else:
        ax = axarr[1][2]
    rej = [
        (p < reject_thresh)
        for p in spot_p_vals
    ]
    plot_slide(
        adata.obs.iloc[kept_inds],
        rej,
        cmap='categorical',
        colorbar=False,
        vmin=None,
        vmax=None,
        title=f'WR P-value < {reject_thresh}',
        ax=ax,
        figure=None,
        ticks=False,
        dsize=dsize,
        row_key=row_key,
        col_key=col_key,
        cat_palette=['#d9d9d9', 'black']
    )
    
    run_regions = [
        ct 
        for ct, p_val in clust_to_p_val.items()
        if p_val >= reject_thresh
    ]
    df_plot = pairwise_clust_between(
        adata,
        [gene_1, gene_2],
        cond_key, 
        run_regions=run_regions, 
        max_perms=max_perms,
        verbose=verbose,
        contrib_thresh=contrib_thresh
    )
    if only_stats:
        ax = axarr[1][1]
    else:
        ax = axarr[1][3]
    res = sns.heatmap(df_plot, cmap='Greys', cbar=False, ax=ax)
    for _, spine in res.spines.items():
        spine.set_visible(True)
    ax.set_title(f'BR P-value < {reject_thresh}')
    ax.set_ylabel('Cluster')
    ax.set_xlabel('Cluster')
    
    #axarr[1][3].set_visible(False)
    
    plt.tight_layout()
    if fig_path is not None:
        fig.savefig(fig_path, format=fig_format, dpi=dpi)


def kernel_diagnostics(
        adata,
        cond_key = 'cluster',
        bandwidth = 5,
        contrib_thresh=10,
        row_key='row',
        col_key='col',
        dsize=12,
        fpath=None,
        fformat='pdf',
        dpi=150
    ):
    """Create plot to visualize the spatial kernel used for SpatialCorr's statistical analyses.

    This function will plot the following analyses:
    Top left: The annotated regions/clusters
    Bottom left: The kernel weights at a randomly chosen spot (i.e., a row of the kernel matrix)
    Top middle: The effective number of samples used to estimate correlation at each spot (i.e., the sum of each row of the kernel matrix)
    Bottom middle: The spots that would be filtered when applying an effective spots threshold of `contrib_thresh` (shown in grey)
    Top right: A distribution of the effective number of samples used to estimate correlation at each spot across the entire slide. The red verticle line shows the effective samples threshold set by `contrib_thresh`

    Parameters
    ----------
    adata : AnnData
        spatial gene expression dataset with spatial coordinates
        stored in `adata.obs`
    cond_key : string
        the name of the column in `adata.obs` storing the cluster
        assignments
    bandwidth : int
        the kernel bandwidth used by the test
    contrib_thresh : int, optional (default: 10)
        threshold for the  total weight of all samples contributing
        to the correlation estimate at each spot. Spots with total
        weight less than this value will be filtered prior to running
        the test
    row_key : string, optional (default: 'row')
        the name of the column in `adata.obs` storing the row coordinates
        of each spot
    col_key : string, optional (default: 'col')
        the name of the column in `adata.obs` storing the column
        coordinates of each spot
    dsize: int, optional (default: 12)
        the size of the dots in the scatterplot
    fpath: string, optional (default: None)
        Path to write figure image.
    fformat: string, {'pdf', 'png'} (default: 'pdf')
        Format of the output figure file.
    dpi: int (default: 150)
        Resolution of output image.

    Returns
    -------
    None
    """
    fig, axarr = plt.subplots(
        2,
        3,
        figsize=(10.5,7)
    )

    # Plot clusters
    plot_slide(
        adata.obs,
        adata.obs[cond_key],
        cmap='categorical',
        colorbar=True,
        title=f'Cluster',
        ax=axarr[0][0],
        figure=None,
        ticks=False,
        dsize=dsize,
        colorticks=None,
        row_key=row_key,
        col_key=col_key
    )

    # Compute kernel matrices
    kernel_matrix_no_cond = compute_kernel_matrix(
        adata.obs,
        bandwidth=bandwidth,
        y_col=row_key,
        x_col=col_key,
        condition_on_region=False,
        region_key=None
    )
    kernel_matrix = compute_kernel_matrix(
        adata.obs,
        bandwidth=bandwidth,
        y_col=row_key,
        x_col=col_key,
        condition_on_region=True,
        region_key=cond_key
    )

    # Plot clusters
    plot_slide(
        adata.obs,
        adata.obs[cond_key],
        cmap='categorical',
        colorbar=True,
        title=f'Clusters',
        ax=axarr[0][0],
        figure=None,
        ticks=False,
        dsize=dsize,
        colorticks=None,
        row_key=row_key,
        col_key=col_key
    )

    # Plot kernel
    rand_index = np.random.randint(
        0, high=kernel_matrix_no_cond.shape[0]
    )
    rand_row = adata.obs.iloc[rand_index][row_key]
    rand_col = adata.obs.iloc[rand_index][col_key]
    plot_slide(
        adata.obs,
        kernel_matrix_no_cond[
            np.random.randint(
                0, high=kernel_matrix_no_cond.shape[0]
            )
        ],
        cmap='viridis',
        colorbar=False,
        title=f'Kernel weights at spot ({rand_row}, {rand_col})',
        ax=axarr[1][0],
        figure=None,
        ticks=False,
        dsize=dsize,
        colorticks=None,
        row_key=row_key,
        col_key=col_key
    )

    # Plot 
    plot_slide(
        adata.obs,
        np.sum(kernel_matrix, axis=1),
        row_key=row_key,
        col_key=col_key,
        ax=axarr[0][1],
        title='Effective samples',
        colorbar=True,
        figure=fig,
        dsize=dsize,
        ticks=False,
        colorticks=None
    )

    # Plot distribution of effective samples
    sns.histplot(
        np.sum(kernel_matrix, axis=1),
        ax=axarr[0][2]
    )
    axarr[0][2].set_title('Distribution of Effective\nSamples')
    axarr[0][2].set_xlabel('Effective Samples')
    axarr[0][2].set_ylabel('No. of Spots')
    axarr[0][2].axvline(x=contrib_thresh, c='r')

    axarr[1][2].set_visible(False)
    
    # Plot which spots are filtered
    plot_filtered_spots(
        adata,
        kernel_matrix,
        contrib_thresh,
        row_key=row_key,
        col_key=col_key,
        ax=axarr[1][1],
        figure=None,
        dsize=dsize,
        ticks=False
    )

    plt.tight_layout()
    if fpath is not None:
        fig.savefig(fpath, format=fformat, dpi=dpi)

def analysis_pipeline_set(
        adata,
        genes,
        cond_key='cluster',
        bandwidth=5,
        max_perms=500,
        row_key='row',
        col_key='col',
        reject_thresh=0.05,
        contrib_thresh=10,
        dsize=12,
        run_br=False,
        spot_to_neighbors=None,
        spot_to_neighbors_clust=None,
        n_procs=5,
        verbose=1,
        fig_path=None,
        fig_format='pdf',
        dpi=150
    ):
    """
    Run a SpatialCorr analysis pipeline on a set of genes.

    This function will run the following analyses:
    1. For each cluster, compute a WR P-value
    2. Remove all clusters with WR P-value < `reject_thresh` for BR-test and for remaining clusters, compute BR P-value testing for differential correlation between the two clusters

    Parameters
    ----------
    genes: List
        List of genes in the gene set
    adata : AnnData
        spatial gene expression dataset with spatial coordinates
        stored in `adata.obs`
    bandwidth : int
        the kernel bandwidth used by the test
    cond_key : string
        the name of the column in `adata.obs` storing the cluster
        assignments
    row_key : string, optional (default: 'row')
        the name of the column in `adata.obs` storing the row coordinates
        of each spot
    col_key : string, optional (default: 'col')
        the name of the column in `adata.obs` storing the column
        coordinates of each spot
    reject_thresh: float (default: 0.05)
        P-value threshold used to reject the null hypothesis for each
        region's WR-test as well as region-pairwise BR-tests.
    dsize: int, optional (default: 12)
        the size of the dots in the scatterplot
    max_perms : int, optional (default: 500)
        Maximum number of permutations to compute for the permutation
        test
    n_procs : int, optional (default: 1)
        number of processes to run in parallel
    verbose : int, optional (default: 1)
        the verbosity. Higher verbosity will lead to more debugging
        information printed to standard output
    contrib_thresh : int, optional (default: 10)
        threshold for the  total weight of all samples contributing
        to the correlation estimate at each spot. Spots with total
        weight less than this value will be filtered prior to running
        the test
    fig_path: string, optional (default: None)
        Path to write figure image.
    fig_format: string, {'pdf', 'png'} (default: 'pdf')
        Format of the output figure file.
    dpi: int (default: 150)
        Resolution of output image.

    Returns
    -------
    None
    """
    fig, axarr = plt.subplots(
        2,
        2,
        figsize=(7,7)
    )

    plot_slide(
        adata.obs,
        adata.obs[cond_key],
        cmap='categorical',
        colorbar=True,
        title=f'Clusters',
        ax=axarr[0][0],
        figure=None,
        ticks=False,
        dsize=dsize,
        colorticks=None,
        row_key=row_key,
        col_key=col_key
    )

    # Plot filtered spots
    kernel_matrix = compute_kernel_matrix(
        adata.obs,
        bandwidth=bandwidth,
        y_col=row_key,
        x_col=col_key,
        condition_on_region=(not cond_key is None),
        region_key=cond_key
    )
    plot_filtered_spots(
        adata,
        kernel_matrix,
        contrib_thresh,
        row_key=row_key,
        col_key=col_key,
        ax=axarr[1][0],
        figure=None,
        dsize=dsize,
        ticks=False
    )

    p_val, additional = run_test(
        adata,
        genes,
        bandwidth,
        cond_key=cond_key,
        contrib_thresh=contrib_thresh,
        row_key=row_key,
        col_key=col_key,
        verbose=verbose,
        n_procs=n_procs,
        run_br=run_br,
        compute_spotwise_pvals=True,
        max_perms=max_perms,
        mc_pvals=False,
        spot_to_neighbors=spot_to_neighbors_clust
    )
    kept_inds = additional['kept_inds']
    clust_to_p_val = additional['region_to_p_val']

    spot_p_vals = [
        clust_to_p_val[ct]
        for ct in adata.obs.iloc[kept_inds][cond_key]
    ]

    rej = [
        (p < reject_thresh)
        for p in spot_p_vals
    ]
    plot_slide(
        adata.obs.iloc[kept_inds],
        rej,
        cmap='categorical',
        colorbar=False,
        vmin=None,
        vmax=None,
        title=f'WR P-value < {reject_thresh}',
        ax=axarr[0][1],
        figure=None,
        ticks=False,
        dsize=dsize,
        row_key=row_key,
        col_key=col_key,
        cat_palette=['#d9d9d9', 'black']
    )

    #plot_slide(
    #    adata.obs.iloc[kept_inds],
    #    -1 * np.log10(spot_p_vals),
    #    cmap='viridis',
    #    colorbar=False,
    #    vmin=0,
    #    vmax=-1 * np.log10(1/max_perms),
    #    title=r'-$log_{10}$ WR P-value',
    #    ax=axarr[0][1],
    #    figure=None,
    #    ticks=False,
    #    dsize=dsize,
    #    colorticks=None,
    #    row_key=row_key,
    #    col_key=col_key
    #)


    run_regions = [
        ct
        for ct, p_val in clust_to_p_val.items()
        if p_val >= 0.05
    ]

    df_plot = pairwise_clust_between(
        adata,
        genes,
        cond_key,
        run_regions=run_regions,
        max_perms=max_perms,
        verbose=verbose,
        contrib_thresh=contrib_thresh
    )

    mask = np.triu(np.ones_like(np.array(df_plot)))
    res = sns.heatmap(df_plot, cmap='Greys', cbar=False, ax=axarr[1][1])#, mask=mask)
    for _, spine in res.spines.items():
        spine.set_visible(True)
    axarr[1][1].set_title('BR P-value < 0.05')
    axarr[1][1].set_ylabel('Cluster')
    axarr[1][1].set_xlabel('Cluster')

    #axarr[1][3].set_visible(False)
    
    plt.tight_layout()
    plt.tight_layout()
    if fig_path is not None:
        fig.savefig(fig_path, format=fig_format, dpi=dpi)


def pairwise_clust_between(
        adata,
        genes, 
        cond_key='cluster', 
        row_key='row', 
        col_key='col', 
        contrib_thresh=10,
        max_perms=500,
        n_procs=5,
        verbose=1,
        run_regions=None
    ):
    ct_to_ct_to_pval, ct_to_ct_to_adj_pval = run_test_between_region_pairs(
        adata,
        genes,
        5,
        cond_key,
        contrib_thresh=contrib_thresh,
        row_key=row_key,
        col_key=col_key,
        verbose=verbose,
        n_procs=n_procs,
        standardize_var=False,
        max_perms=max_perms,
        mc_pvals=True,
        spot_to_neighbors=None,
        run_regions=run_regions
    )
    da = []
    for ct_1 in sorted(set(ct_to_ct_to_pval.keys())):
        row = []
        for ct_2 in sorted(set(ct_to_ct_to_pval.keys())):
            if ct_1 == ct_2:
                row.append(False)
            else:
                row.append(ct_to_ct_to_pval[ct_1][ct_2] < 0.05)
        da.append(row)
    df_plot = pd.DataFrame(
        data=da,
        columns=sorted(set(ct_to_ct_to_pval.keys())),
        index=sorted(set(ct_to_ct_to_pval.keys()))
    )
    return df_plot
