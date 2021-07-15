import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from collections import defaultdict
import seaborn as sns

from . import statistical_test as st
from . import utils

def plot_filtered_spots(
        adata, 
        kernel_matrix, 
        contrib_thresh,
        row_key='row',
        col_key='col',
        ax=None,
        figure=None,
        dsize=37,
        ticks=True,
        colorticks=None
    ):
    # Filter spots with too little contribution
    # from neighbors
    contrib = np.sum(kernel_matrix, axis=1)
    keep_inds = [
        i
        for i, c in enumerate(contrib)
        if c >= contrib_thresh
    ]
    print('Kept {}/{} spots.'.format(len(keep_inds), len(adata.obs)))

    cat = []
    keep_inds = set(keep_inds)
    for ind in range(adata.obs.shape[0]):
        if ind in keep_inds:
            cat.append('Kept')
        else:
            cat.append('Filtered')
    cat_palette = ['grey', 'black']
    plot_slide(
        adata.obs,
        cat,
        cmap='categorical',
        colorbar=False,
        vmin=None,
        vmax=None,
        title='Filtered Spots',
        ax=ax,
        figure=figure,
        ticks=ticks,
        dsize=dsize,
        row_key=row_key,
        col_key=col_key,
        cat_palette=cat_palette
    )


def _estimate_correlations(kernel_matrix, expr_1, expr_2):
    cov_mats = st.kernel_estimation(
        kernel_matrix,
        np.array([expr_1, expr_2])
    )
    covs = [
        c[0][1]
        for c in cov_mats
    ]
    corrs = [
        c[0][1] / np.sqrt(c[0][0] * c[1][1])
        for c in cov_mats
    ]
    return corrs 


def plot_correlation(
        adata, 
        gene_1, 
        gene_2, 
        sigma=5, 
        contrib_thresh=10, 
        kernel_matrix=None, 
        row_key='row', 
        col_key='col', 
        condition=None,
        cmap='RdBu_r',
        colorbar=True,
        ticks=True,
        ax=None,
        figure=None,
        dsize=10,
        estimate='local'
    ):
    if estimate == 'local':
        corrs, keep_inds = _plot_correlation_local(
            adata,
            gene_1,
            gene_2,
            sigma=sigma,
            contrib_thresh=contrib_thresh,
            kernel_matrix=kernel_matrix,
            row_key=row_key, 
            col_key=col_key, 
            condition=condition,
            cmap=cmap,
            colorbar=colorbar,
            ticks=ticks,
            ax=ax,
            figure=figure,
            dsize=dsize
        )
    elif estimate == 'regional':
        corrs, keep_inds = _plot_correlation_regional(
            adata,
            gene_1,
            gene_2,
            condition,
            kernel_matrix=kernel_matrix,
            row_key=row_key,
            col_key=col_key, 
            cmap=cmap,
            colorbar=colorbar,
            ticks=ticks,
            ax=ax,
            figure=figure,
            dsize=dsize
        )
    return corrs, keep_inds


def plot_local_scatter(
        adata, 
        gene_1, 
        gene_2, 
        row, 
        col, 
        plot_vals, 
        color_spots=None, 
        condition=None,
        row_key='row', 
        col_key='col',
        cmap='RdBu_r',
        neighb_color='black'
    ):
    expr_1 = adata.obs_vector(gene_1)
    expr_2 = adata.obs_vector(gene_2)

    meta_df = adata.obs.copy()
    meta_df['tissue'] = [1 for i in range(meta_df.shape[0])]
    row_col_to_barcode = utils.map_row_col_to_barcode(
        meta_df, 
        row_key=row_key, 
        col_key=col_key
    )
    bc_to_neighs, bc_to_above_neighs, bc_to_below_neighs = utils.map_coords_to_neighbors(
        meta_df,
        row_col_to_barcode,
        row_key='row',
        col_key='col'
    )

    if condition is not None:
        bc_to_ct = {
            bc: ro[condition]
            for bc, ro in meta_df.iterrows()
        }

        ct_to_bcs = defaultdict(lambda: [])
        for bc, ro in meta_df.iterrows():
            ct = ro[condition]
            ct_to_bcs[ct].append(bc)

        bc_to_neighs_new = {}
        for bc, neighs in bc_to_neighs.items():
            new_neighs = set(neighs) & set(ct_to_bcs[bc_to_ct[bc]])
            bc_to_neighs_new[bc] = new_neighs
        bc_to_neighs = bc_to_neighs_new

    plot_bc = row_col_to_barcode[row][col]
    barcodes_to_index = {
        bc: index
        for index, bc in enumerate(meta_df.index)
    }
    indices = [barcodes_to_index[bc] for bc in bc_to_neighs[plot_bc]]
    indices.append(barcodes_to_index[plot_bc])

    #if condition is not None:
    #    # Map each group id to its indices
    #    ct_to_inds = defaultdict(lambda: [])
    #    for r_i, (bc, row) in enumerate(meta_df.iterrows()):
    #        ct = row[condition]
    #        ct_to_inds[ct].append(r_i)
#
#        # Filter indices based on group
#        plot_ct = meta_df.loc[plot_bc][condition]
#        keep_inds = ct_to_inds[plot_ct]
#        indices = sorted(set(indices) & set(keep_inds))

    expr = np.array([expr_1, expr_2])
    sample_neigh = expr.T[indices]

    figure, axarr = plt.subplots(
        1,
        2,
        figsize=(10,5)
    )

    plot_neighborhood(
        meta_df,
        [plot_bc],
        bc_to_neighs,
        plot_vals,
        ax=axarr[0],
        dot_size=10,
        vmin=-1,
        vmax=1,
        cmap=cmap,
        neighb_color=neighb_color,
        row_key='row',
        col_key='col'
    )

    if color_spots is not None:
        sns.regplot(
            x=sample_neigh.T[0],
            y=sample_neigh.T[1],
            ax=axarr[1],
            scatter_kws={
                'color': None,
                'c': color_spots[indices],
                'cmap': 'viridis_r',
                'vmin': 0,
                'vmax': 1
            }
        )
    else:
        sns.regplot(
            x=sample_neigh.T[0], 
            y=sample_neigh.T[1], 
            ax=axarr[1]
        )


def _plot_correlation_local(
        adata,
        gene_1,
        gene_2,
        sigma=5,
        contrib_thresh=10,
        kernel_matrix=None,
        row_key='row', 
        col_key='col', 
        condition=None,
        cmap='RdBu_r',
        colorbar=True,
        ticks=True,
        ax=None,
        figure=None,
        dsize=10,
        estimate='local'
    ):
    if kernel_matrix is None:
        kernel_matrix = st._compute_kernel_matrix(
            adata.obs,
            sigma=sigma,
            y_col=row_key,
            x_col=col_key,
            condition_on_cell_type=(not condition is None),
            cell_type_key=condition
        )
        keep_inds = [
            ind
            for ind, contrib in enumerate(np.sum(kernel_matrix, axis=1))
            if contrib >= contrib_thresh
        ]
        kernel_matrix = kernel_matrix[:,keep_inds]
        kernel_matrix = kernel_matrix[keep_inds,:]

    # Filter the spots
    adata = adata[keep_inds,:]

    corrs = _estimate_correlations(
        kernel_matrix, 
        adata.obs_vector(gene_1), 
        adata.obs_vector(gene_2)
    )
    plot_slide(
        adata.obs,
        corrs,
        cmap=cmap,
        colorbar=colorbar,
        vmin=-1,
        vmax=1,
        dsize=dsize,
        row_key=row_key,
        col_key=col_key,
        ticks=ticks,
        ax=ax,
        figure=figure
    )
    return corrs, keep_inds


def _plot_correlation_regional(
        adata,
        gene_1,
        gene_2,
        condition,
        kernel_matrix=None,
        row_key='row',
        col_key='col',
        cmap='RdBu_r',
        colorbar=True,
        ticks=True,
        ax=None,
        figure=None,
        dsize=10
    ):
    expr_1 = adata.obs_vector(gene_1)
    expr_2 = adata.obs_vector(gene_2)
   
    # Map each region ID to the spot indices
    ct_to_indices = defaultdict(lambda: [])
    for r_i, (r_ind, row) in enumerate(adata.obs.iterrows()):
        ct = row[condition]
        ct_to_indices[ct].append(r_i)
            
    ct_to_corr = {
        ct: np.corrcoef([
            np.array(expr_1)[inds],
            np.array(expr_2)[inds]
        ])[0][1]
        for ct, inds in ct_to_indices.items()
    }
    corrs = [
        ct_to_corr[ct]
        for ct in adata.obs[condition]
    ]
    plot_slide(
        adata.obs, 
        corrs, 
        cmap=cmap,
        colorbar=colorbar,
        vmin=-1,
        vmax=1,
        dsize=dsize,
        ticks=ticks,
        ax=ax,
        figure=figure
    )
    keep_inds = list(range(adata.obs.shape[0]))
    return corrs, keep_inds

def plot_slide(
        df,
        values,
        cmap='viridis',
        colorbar=False,
        vmin=None,
        vmax=None,
        title=None,
        ax=None,
        figure=None,
        ticks=True,
        dsize=37,
        colorticks=None,
        row_key='row',
        col_key='col',
        cat_palette=None
    ):
    y = -1 * np.array(df[row_key])
    x = df[col_key]

    if ax is None:
        if colorbar:
            width = 7
        else:
            width = 5
        figure, ax = plt.subplots(
            1,
            1,
            figsize=(width,5)
        )

    if cmap == 'categorical':
        if cat_palette is None:
            pal = PALETTE 
        else:
            pal = cat_palette

        val_to_index = {
            val: ind
            for ind, val in enumerate(sorted(set(values)))
        }
        colors = [
            pal[val_to_index[val]]
            for val in values
        ]
        patches = [
            mpatches.Patch(color=pal[val_to_index[val]], label=val)
            for val in sorted(set(values))
        ]
        ax.scatter(x,y,c=colors, s=dsize)
        ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left',)
    else:
        im = ax.scatter(x,y,c=values, cmap=cmap, s=dsize, vmin=vmin, vmax=vmax)
        if colorbar:
            if vmin is None or vmax is None:
                figure.colorbar(im, ax=ax, ticks=colorticks)
            else:
                figure.colorbar(im, ax=ax, boundaries=np.linspace(vmin,vmax,100), ticks=colorticks)
    if title is not None:
        ax.set_title(title)
    if not ticks:
        ax.set_xticks([])
        ax.set_yticks([])

def plot_neighborhood(
        df,
        sources,
        bc_to_neighbs,
        plot_vals,
        plot=False,
        ax=None,
        keep_inds=None,
        dot_size=30,
        vmin=0,
        vmax=1,
        cmap='RdBu_r',
        ticks=True,
        title=None,
        condition=False,
        cell_type_key=None,
        title_size=12,
        neighb_color='black',
        row_key='row',
        col_key='col'
    ):

    # Get all neighborhood spots
    all_neighbs = set()
    for source in sources:
        neighbs = set(bc_to_neighbs[source])
        if condition:
            ct_spots = set(df.loc[df[cell_type_key] == df.loc[source][cell_type_key]].index)
            neighbs = neighbs & ct_spots
        all_neighbs.update(neighbs)

    if keep_inds is not None:
        all_neighbs &= set(keep_inds)

    y = -1 * np.array(df[row_key])
    x = df[col_key]
    colors=plot_vals
    ax.scatter(x,y,c=colors, s=dot_size, cmap=cmap, vmin=vmin, vmax=vmax)

    colors = []
    plot_inds = []
    for bc_i, bc in enumerate(df.index):
        if bc in sources:
            plot_inds.append(bc_i)
            colors.append(neighb_color)
        elif bc in all_neighbs:
            plot_inds.append(bc_i)
            colors.append(neighb_color)
    if ax is None:
        figure, ax = plt.subplots(
            1,
            1,
            figsize=(5,5)
        )
    y = -1 * np.array(df.iloc[plot_inds][row_key])
    x = df.iloc[plot_inds][col_key]
    ax.scatter(x,y,c=colors, s=dot_size)

    # Re-plot the colored dots over the highlighted neighborhood. Make 
    # the dots smaller so that the highlights stand out.
    colors=np.array(plot_vals)[plot_inds]
    ax.scatter(x,y,c=colors, cmap=cmap, s=dot_size*0.25, vmin=vmin, vmax=vmax)

    if not title:
        ax.set_title(
            'Neighborhood around ({}, {})'.format(
                df.loc[source][row_key],
                df.loc[source][col_key]
            ),
            fontsize=title_size
        )
    else:
        ax.set_title(title, fontsize=title_size)
    if not ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    if plot:
        plt.show()
    return ax

