import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from .statistical_test  import run_test, run_test_between_region_pairs, _compute_kernel_matrix
from .plot import plot_slide, plot_correlation, plot_ci_overlap, plot_filtered_spots

def analysis_pipeline_pair(
        g1, 
        g2,
        adata,
        cond_key,
        row_key='row',
        col_key='col',
        dsize=12,
        sigma=5, 
        max_perms=500, 
        n_procs=5, 
        test_between=False, 
        spot_to_neighbors=None, 
        spot_to_neighbors_clust=None,
        contrib_thresh=10,
        verbose=1
    ):
    fig, axarr = plt.subplots(
        2,
        4,
        figsize=(14,7)
    )  
    
    expr_1 = adata.obs_vector(g1)
    expr_2 = adata.obs_vector(g2)
    min_expr = min(list(expr_1) + list(expr_2))
    max_expr = max(list(expr_1) + list(expr_2))
        
    plot_slide(
        adata.obs,
        expr_1,
        cmap='turbo',
        colorbar=False,
        vmin=min_expr,
        vmax=max_expr,
        title=f'{g1} expression',
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
        cmap='turbo',
        colorbar=False,
        vmin=min_expr,
        vmax=max_expr,
        title=f'{g2} expression',
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
    kernel_matrix = _compute_kernel_matrix(
        adata.obs,
        sigma=sigma,
        y_col=row_key,
        x_col=col_key,
        condition_on_cell_type=(not cond_key is None),
        cell_type_key=cond_key
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
        ticks=False,
        colorticks=None
    )
    
    plot_correlation(
        adata,
        g1,
        g2,
        sigma=sigma,
        contrib_thresh=contrib_thresh,
        row_key=row_key,
        col_key=col_key,
        condition=cond_key,
        cmap='RdBu_r',
        colorbar=False,
        ticks=False,
        ax=axarr[0][2],
        figure=fig,
        dsize=dsize,
        estimate='local',
        title='Correlation\n(kernel estimate)'
    )
    
    plot_ci_overlap(
        g1,
        g2,
        adata,
        cond_key=cond_key,
        kernel_matrix=None,
        sigma=sigma,
        row_key=row_key,
        col_key=col_key,
        title='Regions of non-zero\ncorrelation',
        ax=axarr[0][3],
        figure=None,
        ticks=False,
        dsize=dsize,
        colorticks=None,
        neigh_thresh=10
    )

    p_val, t_obs, t_nulls, kept_inds, obs_spot_lls, spotwise_t_nulls, clust_to_p_val = run_test(
        adata,
        [g1, g2],
        sigma,
        cond_key=cond_key,
        contrib_thresh=10,
        row_key=row_key,
        col_key=col_key,
        verbose=verbose,
        n_procs=n_procs,
        test_between_conds=test_between,
        compute_spotwise_pvals=True,
        max_perms=max_perms,
        mc_pvals=False,
        spot_to_neighbors=spot_to_neighbors_clust
    )
    
    spot_p_vals = [
        clust_to_p_val[ct]
        for ct in adata.obs.iloc[kept_inds][cond_key]
    ]
    
    plot_slide(
        adata.obs.iloc[kept_inds],
        -1 * np.log10(spot_p_vals),
        cmap='viridis',
        colorbar=False,
        vmin=0,
        vmax=-1 * np.log10(1/max_perms),
        title=r'-$log_{10}$ WHR P-value',
        ax=axarr[1][2],
        figure=None,
        ticks=False,
        dsize=dsize,
        colorticks=None,
        row_key=row_key,
        col_key=col_key
    )
    
    run_regions = [
        ct 
        for ct, p_val in clust_to_p_val.items()
        if p_val >= 0.05
    ]
    
    df_plot = pairwise_clust_between(
        [g1, g2], 
        adata, 
        cond_key, 
        run_regions=run_regions, 
        max_perms=max_perms,
        verbose=verbose,
        contrib_thresh=contrib_thresh
    )
    res = sns.heatmap(df_plot, cmap='Greys', cbar=False, ax=axarr[1][3])
    for _, spine in res.spines.items():
        spine.set_visible(True)
    axarr[1][3].set_title('BHR P-value < 0.05')
    axarr[1][3].set_ylabel('Cluster')
    axarr[1][3].set_xlabel('Cluster')
    
    #axarr[1][3].set_visible(False)
    
    plt.tight_layout()

def pairwise_clust_between(
        genes, 
        adata, 
        cond_key, 
        row_key='row', 
        col_key='col', 
        max_perms=500,
        n_procs=5,
        run_regions=None,
        verbose=1,
        contrib_thresh=10
    ):
    ct_to_ct_to_pval = run_test_between_region_pairs(
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
