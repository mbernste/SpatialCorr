import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import math
from collections import defaultdict
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, set_link_color_palette
from sklearn.cluster import AgglomerativeClustering

from . import statistical_test as st
from . import utils


PALETTE_MANY = [
    "#0652ff", #  electric blue
    "#e50000", #  red
    "#9a0eea", #  violet
    "#01b44c", #  shamrock
    "#fedf08", #  dandelion
    "#00ffff", #  cyan
    "#89fe05", #  lime green
    "#a2cffe", #  baby blue
    "#dbb40c", #  gold
    "#029386", #  teal
    "#ff9408", #  tangerine
    "#d8dcd6", #  light grey
    "#80f9ad", #  seafoam
    "#3d1c02", #  chocolate
    "#fffd74", #  butter yellow
    "#536267", #  gunmetal
    "#f6cefc", #  very light purple
    "#650021", #  maroon
    "#020035", #  midnight blue
    "#b0dd16", #  yellowish green
    "#9d7651", #  mocha
    "#c20078", #  magenta
    "#380282", #  indigo
    "#ff796c", #  salmon
    "#874c62"  #  dark muave
]


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
    cat_palette = ['#595959', '#d9d9d9']
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
        corr_magnitude=False,
        cmap='RdBu_r',
        colorbar=True,
        ticks=True,
        ax=None,
        figure=None,
        dsize=10,
        estimate='local',
        title=None,
        spot_borders=False,
        border_color='black',
        border_size=0.3
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
            corr_magnitude=corr_magnitude,
            colorbar=colorbar,
            ticks=ticks,
            ax=ax,
            figure=figure,
            dsize=dsize,
            title=title,
            spot_borders=spot_borders,
            border_color=border_color,
            border_size=border_size
        )
        extra_data = {}
    elif estimate == 'regional':
        corrs, keep_inds, ct_to_corr = _plot_correlation_regional(
            adata,
            gene_1,
            gene_2,
            condition,
            kernel_matrix=kernel_matrix,
            row_key=row_key,
            col_key=col_key, 
            cmap=cmap,
            corr_magnitude=corr_magnitude,
            colorbar=colorbar,
            ticks=ticks,
            ax=ax,
            figure=figure,
            dsize=dsize,
            title=title,
            spot_borders=spot_borders,
            border_color=border_color,
            border_size=border_size
        )
        extra_data={'region_to_corr': ct_to_corr}
    return corrs, keep_inds, extra_data


def plot_ci_overlap(
        gene_1,
        gene_2, 
        adata,
        cond_key,
        kernel_matrix=None,
        bandwidth=5,
        row_key='row',
        col_key='col',
        title=None,
        ax=None,
        figure=None,
        ticks=False,
        dsize=12,
        colorticks=None,
        neigh_thresh=10
    ):
    if kernel_matrix is None:
        kernel_matrix = st._compute_kernel_matrix(
            adata.obs,
            sigma=bandwidth,
            y_col=row_key,
            x_col=col_key,
            condition_on_cell_type=(not cond_key is None),
            cell_type_key=cond_key
        )

    # Compute confidence intervals
    row_col_to_barcode = utils.map_row_col_to_barcode(
        adata.obs,
        row_key=row_key,
        col_key=col_key
    )
    bc_to_neighs = bc_to_neighs = utils.compute_neighbors(
        adata.obs,
        row_col_to_barcode,
        row_key=row_key,
        col_key=col_key
    )
    cis, keep_inds = st.est_corr_cis(
        gene_1, gene_2,
        adata,
        bandwidth=bandwidth,
        precomputed_kernel=kernel_matrix,
        cond_key=cond_key,
        neigh_thresh=neigh_thresh,
        spot_to_neighs=bc_to_neighs,
        n_boots=100
    )

    # Compute spotwise labels
    bin_corrs = []
    for ci in cis:
        if -1 * 0 > ci[1]:
            bin_corrs.append(-1)
        elif 0 < ci[0]:
            bin_corrs.append(1)
        else:
            bin_corrs.append(0)

    # Plot slide
    plot_slide(
        adata.obs.iloc[keep_inds],
        bin_corrs,
        cmap='RdBu_r',
        colorbar=False,
        vmin=-1.8,
        vmax=1.8,
        title=title,
        ax=ax,
        figure=figure,
        ticks=False,
        dsize=dsize,
        colorticks=None,
        row_key=row_key,
        col_key=col_key
    )


def plot_local_scatter(
        adata, 
        gene_1, 
        gene_2, 
        row, 
        col, 
        plot_vals, 
        color_spots=None, 
        condition=None,
        vmin=None,
        vmax=None,
        row_key='row', 
        col_key='col',
        cmap='RdBu_r',
        neighb_color='black',
        plot_neigh=True,
        width=10,
        height=5,
        line_color='black',
        scatter_xlim=None,
        scatter_ylim=None,
        scatter_xlabel=None,
        scatter_ylabel=None,
        scatter_title=None,
        fig_path=None,
        fig_format='pdf',
        fig_dpi=150,
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
    bc_to_neighs = utils.compute_neighbors(
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

    expr = np.array([expr_1, expr_2])
    sample_neigh = expr.T[indices]

    if plot_neigh:
        figure, axarr = plt.subplots(
            1,
            2,
            figsize=(width,height)
        )
    else:
        figure, axarr = plt.subplots(
            1,
            1,
            figsize=(width,height)
        )

    if plot_neigh:
        plot_neighborhood(
            meta_df,
            [plot_bc],
            bc_to_neighs,
            plot_vals,
            ax=axarr[0],
            dot_size=10,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            neighb_color=neighb_color,
            row_key=row_key,
            col_key=col_key
        )
    
    if plot_neigh:
        ax = axarr[1]
    else:
        ax = axarr

    if color_spots is not None:
        sns.regplot(
            x=sample_neigh.T[0],
            y=sample_neigh.T[1],
            ax=ax,
            scatter_kws={
                'color': None,
                'c': color_spots[indices],
                'cmap': 'viridis_r',
                'vmin': 0,
                'vmax': 1
            },
            line_kws={"color": line_color}
        )
    else:
        sns.regplot(
            x=sample_neigh.T[0], 
            y=sample_neigh.T[1], 
            ax=ax,
            scatter_kws={
                'color': line_color,
                'cmap': 'viridis_r',
                'vmin': 0,
                'vmax': 1
            },
            line_kws={
                "color": line_color
            }
        )

    if scatter_xlabel:
        ax.set_xlabel(scatter_xlabel)
    else:
        ax.set_xlabel(f'{gene_1} Expression')

    if scatter_ylabel:
        ax.set_ylabel(scatter_ylabel)
    else:
        ax.set_ylabel(f'{gene_2} Expression')

    if scatter_xlim is not None:
        ax.set_xlim(scatter_xlim)
    if scatter_ylim is not None:
        ax.set_ylim(scatter_ylim)
    if scatter_title is not None:
        ax.set_title(scatter_title)

    if fig_path:
        plt.tight_layout()
        figure.savefig(
            fig_path,
            format=fig_format,
            dpi=fig_dpi
        )
        plt.show()


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
        corr_magnitude=False,
        cmap='RdBu_r',
        colorbar=True,
        ticks=True,
        ax=None,
        figure=None,
        dsize=10,
        estimate='local',
        title=None,
        spot_borders=True,
        border_color='black',
        border_size=0.3
    ):
    corrs, keep_inds = utils.compute_local_correlation(
        adata, 
        gene_1,
        gene_2,
        row_key=row_key, 
        col_key=col_key, 
        kernel_matrix=kernel_matrix, 
        condition=condition, 
        sigma=sigma,
        contrib_thresh=contrib_thresh
    )

    if corr_magnitude:
        corrs = np.absolute(corrs)
        vmin = 0
        vmax = 1
    else:
        vmin = -1
        vmax = 1
    plot_slide(
        adata.obs.iloc[keep_inds],
        corrs,
        cmap=cmap,
        colorbar=colorbar,
        vmin=vmin,
        vmax=vmax,
        dsize=dsize,
        row_key=row_key,
        col_key=col_key,
        ticks=ticks,
        ax=ax,
        figure=figure,
        title=title,
        spot_borders=spot_borders,
        border_color=border_color,
        border_size=border_size
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
        corr_magnitude=False,
        cmap='RdBu_r',
        colorbar=True,
        ticks=True,
        ax=None,
        figure=None,
        dsize=10,
        title=None,
        spot_borders=False,
        border_color='black',
        border_size=0.3
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
    corrs = np.array([
        ct_to_corr[ct]
        for ct in adata.obs[condition]
    ])
    if corr_magnitude:
        corrs = np.absolute(corrs)
        vmin = 0
        vmax = 1
    else:
        vmin = -1
        vmax = 1
    plot_slide(
        adata.obs, 
        corrs, 
        cmap=cmap,
        colorbar=colorbar,
        vmin=vmin,
        vmax=vmax,
        dsize=dsize,
        ticks=ticks,
        ax=ax,
        figure=figure,
        title=title,
        spot_borders=spot_borders,
        border_color=border_color,
        border_size=border_size
    )
    keep_inds = list(range(adata.obs.shape[0]))
    return corrs, keep_inds, ct_to_corr


def _plot_slide_one_color(
        df,
        color,
        row_key='row',
        col_key='col',
        dsize=37,
        ax=None
    ):
    if ax is None:
        figure, ax = plt.subplots(
            1,
            1,
            figsize=(5,5)
        )
    y = -1 * np.array(df[row_key])
    x = df[col_key]
    ax.scatter(x,y,c=color, s=dsize)


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
        cat_palette=None,
        spot_borders=False,
        border_color='black',
        border_size=0.3
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

    #if spot_borders:
    #    if border_size is None:
    #        border_size = dsize+5
    #    _plot_slide_one_color(
    #        df,
    #        border_color,
    #        row_key=row_key,
    #        col_key=col_key,
    #        dsize=border_size,
    #        ax=ax
    #    )
    
    if cmap == 'categorical':
        if cat_palette is None:
            pal = PALETTE_MANY 
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
        if spot_borders:
            ax.scatter(x,y,c=colors, s=dsize, edgecolors=border_color, linewidths=border_size)
        else:
            ax.scatter(x,y,c=colors, s=dsize)
        if colorbar:
            ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left',)
    else:
        if spot_borders:
            im = ax.scatter(x,y,c=values, cmap=cmap, s=dsize, vmin=vmin, vmax=vmax, edgecolors=border_color, linewidths=border_size)
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




def mult_genes_plot_correlation(
        plot_genes,
        adata,
        cond_key,
        bandwidth=5,
        precomputed_kernel=None,
        contrib_thresh=10,
        estimate_type='local',
        row_key='row',
        col_key='col',
        dsize=7,
        fig_path=None,
        fig_format='png',
        fig_dpi=150
    ):

    condition = cond_key is not None
    if precomputed_kernel is None:
        kernel_matrix = st._compute_kernel_matrix(
            adata.obs,
            sigma=bandwidth,
            cell_type_key=cond_key,
            condition_on_cell_type=condition,
            y_col=row_key,
            x_col=col_key
        )
    else:
        kernel_matrix = precomputed_kernel

    # Select all genes that are in the data
    plot_genes = [
        gene for gene in plot_genes
        if gene in adata.var.index
    ]

    fig, axarr = plt.subplots(
        len(plot_genes),
        len(plot_genes),
        figsize=(2*len(plot_genes),2*len(plot_genes))
    )

    # Compute kept indices
    corrs, keep_inds = utils.compute_local_correlation(
        adata,
        plot_genes[0], plot_genes[1],
        kernel_matrix=kernel_matrix,
        row_key=row_key,
        col_key=col_key,
        condition=cond_key,
        sigma=bandwidth,
        contrib_thresh=contrib_thresh
    )

    # Filter kernel matrix, if it's provided
    kernel_matrix = kernel_matrix[keep_inds,:]
    kernel_matrix = kernel_matrix[:,keep_inds]

    # Get range of expression values for colormap
    # of expression
    all_expr = []
    for gene in plot_genes:
        expr = adata[keep_inds,:].obs_vector(gene)
        all_expr += list(expr)
    min_expr = min(all_expr)
    max_expr = max(all_expr)

    for row, ax_row in enumerate(axarr):
        for col, ax in enumerate(ax_row):
            gene_1 = plot_genes[row]
            gene_2 = plot_genes[col]

            if row == 0:
                title = gene_2
            else:
                title = None

            if col == row:
                plot_slide(
                    adata[keep_inds,:].obs,
                    adata[keep_inds,:].obs_vector(gene_1),
                    cmap='turbo',
                    title=title,
                    dsize=dsize,
                    ax=ax,
                    figure=fig,
                    ticks=False,
                    vmin=min_expr,
                    vmax=max_expr,
                    row_key=row_key,
                    col_key=col_key
                )
                ax.set_ylabel(gene_1, fontsize=13)
            elif col > row:
                if estimate_type in ['local', 'regional']:
                    corrs, kept_inds, _ = plot_correlation(
                        adata[keep_inds,:],
                        gene_1, gene_2,
                        sigma=bandwidth,
                        contrib_thresh=contrib_thresh,
                        kernel_matrix=kernel_matrix,
                        row_key=row_key,
                        col_key=col_key,
                        condition=cond_key,
                        cmap='RdBu_r',
                        colorbar=False,
                        ticks=False,
                        ax=ax,
                        figure=None,
                        estimate=estimate_type,
                        dsize=dsize,
                        title=title
                    )
                elif estimate_type == 'local_ci':
                    plot_ci_overlap(
                        gene_1,
                        gene_2,
                        adata,
                        cond_key,
                        kernel_matrix=None,
                        bandwidth=bandwidth,
                        row_key=row_key,
                        col_key=col_key,
                        title=None,
                        ax=ax,
                        figure=None,
                        ticks=False,
                        dsize=dsize,
                        colorticks=None,
                        neigh_thresh=contrib_thresh
                    )
            else:
                ax.set_visible(False)

    if fig_path:
        plt.tight_layout()
        fig.savefig(
            fig_path,
            format=fig_format,
            dpi=fig_dpi
        )
        plt.show()


def _compute_pairwise_corrs(
        gene_pairs, 
        adata, 
        cond_key, 
        sigma=5, 
        row_key='row', 
        col_key='col'
    ):
    gps = []
    all_corrs = []
    for g1, g2 in gene_pairs:
        corrs, keep_inds = utils.compute_local_correlation(
            adata, 
            g1, g2,
            kernel_matrix=None, 
            row_key=row_key, 
            col_key=col_key, 
            condition=cond_key, 
            sigma=sigma
        )
        gps.append((g1, g2))
        all_corrs.append(corrs)    
    return all_corrs


def cluster_pairwise_correlations( 
        plot_genes,
        adata,
        cond_key,
        sigma=5,
        row_key='row',
        col_key='col',
        color_thresh=19,
        title=None,
        remove_y_ticks=False,
        fig_path=None,
        fig_format='png',
        fig_dpi=150
    ):

    gene_pairs = []
    for g1_i, g1 in enumerate(plot_genes):
        for g2_i, g2 in enumerate(plot_genes):
            if g1_i >= g2_i:
                continue
            gene_pairs.append((g1, g2))
    gene_pairs = [
        tuple(sorted(x)) 
        for x in gene_pairs
    ]

    all_corrs = _compute_pairwise_corrs(
        gene_pairs,
        adata,
        cond_key,
        sigma=5,
        row_key='row',
        col_key='col'
    )

    pal = list(sns.color_palette("Set2").as_hex())

    def plot_dendrogram(model, **kwargs):
        # Create linkage matrix and then plot the dendrogram

        # create the counts of samples under each node
        counts = np.zeros(model.children_.shape[0])
        n_samples = len(model.labels_)
        for i, merge in enumerate(model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1  # leaf node
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        linkage_matrix = np.column_stack([
            model.children_, 
            model.distances_,
            counts
        ]).astype(float)

        # Plot the corresponding dendrogram
        dendrogram(linkage_matrix, **kwargs)

    fig, ax = plt.subplots(
        1,
        1,
        figsize=(6,4)
    )
    # Setting distance_threshold=0 ensures we compute the full tree.
    model = AgglomerativeClustering(
        distance_threshold=0, 
        n_clusters=None
    )

    model = model.fit(np.array(all_corrs).squeeze())
    
    set_link_color_palette(pal)

    plot_dendrogram(
        model, 
        truncate_mode='level', 
        p=50, 
        labels=[', '.join(x) for x in gene_pairs], 
        color_threshold=color_thresh, 
        leaf_rotation=90, 
        ax=ax, 
        above_threshold_color='grey'
    )

    if title:
        ax.set_title(title)

    if remove_y_ticks:
        ax.set_yticklabels([])
        ax.set_yticks([])


    plt.tight_layout()
    plt.show()
    if fig_path:
        plt.tight_layout()
        fig.savefig(
            fig_path,
            format=fig_format,
            dpi=fig_dpi
        )
        plt.show()

def plot_cluster_scatter(
        gene_1, 
        gene_2, 
        adata, 
        cond_key,
        clust, 
        col_vals=None, 
        cmap=None, 
        color=None, 
        xlim=None, 
        ylim=None, 
        ax=None,
        xlabel=None,
        ylabel=None
    ):
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(3, 3)) 
    ct_to_inds = defaultdict(lambda: [])
    for ind, ct in enumerate(adata.obs[cond_key]):
        ct_to_inds[ct].append(ind)
    expr_1 = adata.obs_vector(gene_1)[ct_to_inds[clust]]
    expr_2 = adata.obs_vector(gene_2)[ct_to_inds[clust]]
    if col_vals is not None:
        scatter_kws = {
            'color': None,
            'c': col_vals[ct_to_inds[clust]],
            'cmap': cmap,
            'alpha': 1.0
        }
        line_kws = {
            'color': col_vals[ct_to_inds[clust]]
        }
        sns.regplot(
            expr_1, 
            expr_2, 
            scatter_kws=scatter_kws, 
            line_kws=line_kws, 
            ax=ax
        ) 
    elif color is not None:
        scatter_kws = {
            'color': color,
            's': 5
        }
        line_kws = {
            'color': color
        }
        sns.regplot(
            expr_1, 
            expr_2, 
            scatter_kws=scatter_kws, 
            line_kws=line_kws, 
            ax=ax
        ) 
    else:
        sns.regplot(expr_1, expr_2)#, s=4)

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)


def cluster_pairwise_scatterplots(
        gene_1,
        gene_2,
        adata,
        cond_key,
        sigma=5,
        row_key='row',
        col_key='col',
        xlim=None,
        ylim=None,
        fig_path=None,
        fig_format='png',
        fig_dpi=150
    ):

    clusts = sorted(set(adata.obs[cond_key]))

    n_cols = min([len(clusts), 5])
    n_rows = math.ceil(len(clusts) / n_cols)

    fig, axarr = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2*n_cols, 2*n_rows)
    )

    ax_r = 0
    ax_c = 0
    for c_i, ct in enumerate(clusts):
        if n_rows > 1:
            ax = axarr[ax_r][ax_c]
        else:
            ax = axarr[ax_c]

        ylabel = None
        xlabel = None
        if ax_c == 0:
            ylabel = f'{gene_2} Expression'
        if ax_r == n_rows-1:
            xlabel = f'{gene_1} Expression'

        plot_cluster_scatter(
            gene_1, 
            gene_2, 
            adata, 
            cond_key,
            ct,
            ax=ax,
            col_vals=None, 
            cmap=None, 
            color=PALETTE_MANY[c_i], 
            xlim=xlim, 
            ylim=ylim,
            xlabel=xlabel,
            ylabel=ylabel
        )

        ax_c += 1
        if ax_c >= n_cols:
            ax_c = 0
            ax_r += 1

        ax.set_title(ct)

    if ax_r < len(axarr)-1:
        for ax_c in range(ax_c, n_cols):
            axarr[ax_r][ax_c].set_visible(False)
            axarr[ax_r - 1][ax_c].set_xlabel(f'{gene_1} Expression')
    plt.tight_layout()
    
    # Save figure
    if fig_path:
        plt.tight_layout()
        fig.savefig(
            fig_path,
            format=fig_format,
            dpi=fig_dpi
        )
        plt.show()
