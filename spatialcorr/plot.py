import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

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

