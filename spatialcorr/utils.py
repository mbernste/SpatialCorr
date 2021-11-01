from collections import defaultdict
import numpy as np

from . import statistical_test as st

def compute_local_correlation(
        adata, 
        gene_1,
        gene_2,
        kernel_matrix=None, 
        row_key='row', 
        col_key='col', 
        condition=None, 
        sigma=5,
        contrib_thresh=10
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

    corrs = np.array(_estimate_correlations(
        kernel_matrix,
        adata.obs_vector(gene_1),
        adata.obs_vector(gene_2)
    ))
    return corrs, keep_inds


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


def map_row_col_to_barcode(df, row_key='row', col_key='col'):
    """
    Map each row, column pair to its barcode.
    """
    row_to_col_to_bc = defaultdict(lambda: {})
    for ri, row in df.iterrows():
        r = row[row_key]
        c = row[col_key]
        b = row.name
        row_to_col_to_bc[r][c] = b
    return row_to_col_to_bc


def permute_coords(
        coords,
        ct_to_indices
    ):
    perms = np.zeros(len(coords))
    for ct, indices in ct_to_indices.items():
        ct_coords = coords[indices]
        ct_perm = np.random.permutation(ct_coords)
        perms[indices] = ct_perm
    return perms


def _add_neighbor(
        curr_bc, 
        c, 
        r, 
        dc, 
        dr, 
        row_col_to_barcode, 
        df, 
        dicts, 
        tissue_key='tissue', 
        row_key='row', 
        col_key='col'
    ):
    """
    c: current column
    r: current row
    dc: column displacement
    dr: row displacement
    """
    dims = (max(df[row_key]), max(df[col_key]))
    if c+dc < dims[1] and c+dc >= 0 and r+dr < dims[0] and r+dr >= 0:
        try:
            bc_cand = row_col_to_barcode[r+dr][c+dc]
            for d in dicts:
                d[curr_bc].append(bc_cand)
        except KeyError:
            pass


def compute_neighbors(
        df, 
        row_col_to_barcode, 
        rad=2, 
        tissue_key='tissue', 
        row_key='row', 
        col_key='col'
    ):
    """
    Map each barcode to its neighbors.
    """
    bc_to_neighs = defaultdict(lambda: [])
    bc_to_above_neighs = defaultdict(lambda: [])
    bc_to_below_neighs = defaultdict(lambda: [])
    df_ts = df # TODO this step is not necessary
    #df_ts = df.loc[df[tissue_key] == 1]
    for ri, row in df_ts.iterrows():
        r = row[row_key]
        c = row[col_key]
        curr_bc = row.name
        # Check left
        #
        #         . . 
        #        * * . 
        #         . .  
        #
        _add_neighbor(
            curr_bc, 
            c, 
            r, 
            -2, 
            0, 
            row_col_to_barcode, 
            df, 
            [bc_to_neighs, bc_to_above_neighs, bc_to_below_neighs],
            tissue_key=tissue_key,
            row_key=row_key,
            col_key=col_key
        )
        # Check right
        #
        #         . . 
        #        . * * 
        #         . .  
        #
        _add_neighbor(
            curr_bc, 
            c, 
            r, 
            +2, 
            0, 
            row_col_to_barcode, 
            df, 
            [bc_to_neighs, bc_to_above_neighs, bc_to_below_neighs],
            tissue_key=tissue_key,
            row_key=row_key,
            col_key=col_key
        )
        # Check up, left
        #
        #         * .
        #        . * .
        #         . .
        #
        _add_neighbor(
            curr_bc, 
            c, 
            r, 
            -1, 
            1, 
            row_col_to_barcode, 
            df, 
            [bc_to_neighs, bc_to_above_neighs],
            tissue_key=tissue_key,
            row_key=row_key,
            col_key=col_key
        )
        # Check up, right
        #
        #         . *
        #        . * .
        #         . .
        #
        _add_neighbor(
            curr_bc, 
            c, 
            r, 
            1, 
            1, 
            row_col_to_barcode, 
            df, 
            [bc_to_neighs, bc_to_above_neighs],
            tissue_key=tissue_key,
            row_key=row_key,
            col_key=col_key
        )
        # Check down, left
        #
        #         . .
        #        . * .
        #         * .
        #
        _add_neighbor(
            curr_bc, 
            c, 
            r, 
            -1, 
            -1, 
            row_col_to_barcode, 
            df, 
            [bc_to_neighs, bc_to_below_neighs],
            tissue_key=tissue_key,
            row_key=row_key,
            col_key=col_key
        )
        # Check down, right
        #
        #         . .
        #        . * .
        #         . *
        #
        _add_neighbor(
            curr_bc, 
            c, 
            r, 
            1, 
            -1, 
            row_col_to_barcode, 
            df, 
            [bc_to_neighs, bc_to_below_neighs],
            tissue_key=tissue_key,
            row_key=row_key,
            col_key=col_key
        )
        if rad > 1:
            # Check two left
            #
            #          . . .
            #         . . . .
            #        * . * . .
            #         . . . .
            #          . . .
            #
            _add_neighbor(
                curr_bc, 
                c, 
                r, 
                -4, 
                0, 
                row_col_to_barcode, 
                df, 
                [bc_to_neighs, bc_to_below_neighs, bc_to_above_neighs],
                tissue_key=tissue_key,
                row_key=row_key,
                col_key=col_key
            )
            # Check two right
            #
            #          . . .
            #         . . . .
            #        . . * . *
            #         . . . .
            #          . . .
            #
            _add_neighbor(
                curr_bc, 
                c, 
                r, 
                4, 
                0, 
                row_col_to_barcode, 
                df, 
                [bc_to_neighs, bc_to_below_neighs, bc_to_above_neighs],
                tissue_key=tissue_key,
                row_key=row_key,
                col_key=col_key
            )
            # Check directly up 2
            #
            #          . * .
            #         . . . .
            #        . . * . .
            #         . . . .
            #          . . .
            #
            _add_neighbor(
                curr_bc, 
                c, 
                r, 
                0, 
                2, 
                row_col_to_barcode, 
                df, 
                [bc_to_neighs, bc_to_above_neighs],
                tissue_key=tissue_key,
                row_key=row_key,
                col_key=col_key
            )
            # Check up 2, left 1
            #
            #          * . .
            #         . . . .
            #        . . * . .
            #         . . . .
            #          . . .
            #
            _add_neighbor(
                curr_bc, 
                c, 
                r, 
                -2, 
                2, 
                row_col_to_barcode, 
                df, 
                [bc_to_neighs, bc_to_above_neighs],
                tissue_key=tissue_key,
                row_key=row_key,
                col_key=col_key
            )
            # Check up 2, right 1
            #
            #          . . *
            #         . . . .
            #        . . * . .
            #         . . . .
            #          . . .
            #
            _add_neighbor(
                curr_bc, 
                c, 
                r, 
                2, 
                2, 
                row_col_to_barcode, 
                df, 
                [bc_to_neighs, bc_to_above_neighs],
                tissue_key=tissue_key,
                row_key=row_key,
                col_key=col_key
            )
            # Check up 1, left 2
            #
            #          . . .
            #         * . . .
            #        . . * . .
            #         . . . .
            #          . . .
            #
            _add_neighbor(
                curr_bc, 
                c, 
                r, 
                -3, 
                1, 
                row_col_to_barcode, 
                df, 
                [bc_to_neighs, bc_to_above_neighs],
                tissue_key=tissue_key,
                row_key=row_key,
                col_key=col_key
            )
            # Check up 1, right 2
            #
            #          . . .
            #         . . . *
            #        . . * . .
            #         . . . .
            #          . . .
            #
            _add_neighbor(
                curr_bc, 
                c, 
                r, 
                3, 
                1, 
                row_col_to_barcode, 
                df, 
                [bc_to_neighs, bc_to_above_neighs],
                tissue_key=tissue_key,
                row_key=row_key,
                col_key=col_key
            )
            # Check directly down 2
            #
            #          . . .
            #         . . . .
            #        . . * . .
            #         . . . .
            #          . * .
            #
            _add_neighbor(
                curr_bc, 
                c, 
                r, 
                0, 
                -2, 
                row_col_to_barcode, 
                df, 
                [bc_to_neighs, bc_to_below_neighs],
                tissue_key=tissue_key,
                row_key=row_key,
                col_key=col_key
            )
            # Check down 2, left 1
            #
            #          . . .
            #         . . . .
            #        . . * . .
            #         . . . .
            #          * . .
            #
            _add_neighbor(
                curr_bc, 
                c, 
                r, 
                -2, 
                -2, 
                row_col_to_barcode, 
                df, 
                [bc_to_neighs, bc_to_below_neighs],
                tissue_key=tissue_key,
                row_key=row_key,
                col_key=col_key
            )
            # Check down 2, right 1
            #
            #          . . .
            #         . . . .
            #        . . * . .
            #         . . . .
            #          . . *
            #
            _add_neighbor(
                curr_bc, 
                c, 
                r, 
                2, 
                -2, 
                row_col_to_barcode, 
                df, 
                [bc_to_neighs, bc_to_below_neighs],
                tissue_key=tissue_key,
                row_key=row_key,
                col_key=col_key
            )
            # Check down 1, left 2
            #
            #          . . .
            #         . . . .
            #        . . * . .
            #         * . . .
            #          . . .
            #
            _add_neighbor(
                curr_bc, 
                c, 
                r, 
                -3, 
                -1, 
                row_col_to_barcode, 
                df, 
                [bc_to_neighs, bc_to_below_neighs],
                tissue_key=tissue_key,
                row_key=row_key,
                col_key=col_key
            )
            # Check down 1, right 2
            #
            #          . . .
            #         . . . .
            #        . . * . .
            #         . . . *
            #          . . .
            #
            _add_neighbor(
                curr_bc, 
                c, 
                r, 
                3, 
                -1, 
                row_col_to_barcode, 
                df, 
                [bc_to_neighs, bc_to_below_neighs],
                tissue_key=tissue_key,
                row_key=row_key,
                col_key=col_key
            )
        if rad > 2:
            # Check three left
            #
            #         . . . .
            #        . . . . . 
            #       . . . . . .
            #      * . . * . . .
            #       . . . . . .
            #        . . . . .
            #         . . . .
            #
            _add_neighbor(
                curr_bc, 
                c, 
                r, 
                -6, 
                0, 
                row_col_to_barcode, 
                df, 
                [bc_to_neighs, bc_to_below_neighs, bc_to_above_neighs],
                tissue_key=tissue_key,
                row_key=row_key,
                col_key=col_key
            )
            #         . . . .
            #        . . . . .
            #       * . . . . .
            #      . . . * . . .
            #       . . . . . .
            #        . . . . .
            #         . . . .
            #
            _add_neighbor(
                curr_bc, 
                c, 
                r, 
                -5, 
                1, 
                row_col_to_barcode, 
                df, 
                [bc_to_neighs, bc_to_above_neighs],
                tissue_key=tissue_key,
                row_key=row_key,
                col_key=col_key
            )
            #         . . . .
            #        * . . . .
            #       . . . . . .
            #      . . . * . . .
            #       . . . . . .
            #        . . . . .
            #         . . . .
            #
            _add_neighbor(
                curr_bc, 
                c, 
                r, 
                -4, 
                2, 
                row_col_to_barcode, 
                df, 
                [bc_to_neighs, bc_to_above_neighs],
                tissue_key=tissue_key,
                row_key=row_key,
                col_key=col_key
            )
            #         * . . .
            #        . . . . .
            #       . . . . . .
            #      . . . * . . .
            #       . . . . . .
            #        . . . . .
            #         . . . .
            #
            _add_neighbor(
                curr_bc, 
                c, 
                r, 
                -3, 
                3, 
                row_col_to_barcode, 
                df, 
                [bc_to_neighs, bc_to_above_neighs],
                tissue_key=tissue_key,
                row_key=row_key,
                col_key=col_key
            )
            #         . * . .
            #        . . . . .
            #       . . . . . .
            #      . . . * . . .
            #       . . . . . .
            #        . . . . .
            #         . . . .
            #
            _add_neighbor(
                curr_bc, 
                c, 
                r, 
                -1, 
                3, 
                row_col_to_barcode, 
                df, 
                [bc_to_neighs, bc_to_above_neighs],
                tissue_key=tissue_key,
                row_key=row_key,
                col_key=col_key
            )
            #         . . * .
            #        . . . . .
            #       . . . . . .
            #      . . . * . . .
            #       . . . . . .
            #        . . . . .
            #         . . . .
            #
            _add_neighbor(
                curr_bc, 
                c, 
                r, 
                1, 
                3, 
                row_col_to_barcode, 
                df, 
                [bc_to_neighs, bc_to_above_neighs],
                tissue_key=tissue_key,
                row_key=row_key,
                col_key=col_key
            )
            #         . . . *
            #        . . . . .
            #       . . . . . .
            #      . . . * . . .
            #       . . . . . .
            #        . . . . .
            #         . . . .
            #
            _add_neighbor(
                curr_bc, 
                c, 
                r, 
                3, 
                3, 
                row_col_to_barcode, 
                df, 
                [bc_to_neighs, bc_to_above_neighs],
                tissue_key=tissue_key,
                row_key=row_key,
                col_key=col_key
            )
            #         . . . .
            #        . . . . *
            #       . . . . . .
            #      . . . * . . .
            #       . . . . . .
            #        . . . . .
            #         . . . .
            #
            _add_neighbor(
                curr_bc, 
                c, 
                r, 
                4, 
                2, 
                row_col_to_barcode, 
                df, 
                [bc_to_neighs, bc_to_above_neighs],
                tissue_key=tissue_key,
                row_key=row_key,
                col_key=col_key
            )
            #         . . . .
            #        . . . . .
            #       . . . . . *
            #      . . . * . . .
            #       . . . . . .
            #        . . . . .
            #         . . . .
            #
            _add_neighbor(
                curr_bc, 
                c, 
                r, 
                5, 
                1, 
                row_col_to_barcode, 
                df, 
                [bc_to_neighs, bc_to_above_neighs],
                tissue_key=tissue_key,
                row_key=row_key,
                col_key=col_key
            )
            #         . . . .
            #        . . . . .
            #       . . . . . .
            #      . . . * . . *
            #       . . . . . .
            #        . . . . .
            #         . . . .
            #
            _add_neighbor(
                curr_bc, 
                c, 
                r, 
                6, 
                0, 
                row_col_to_barcode, 
                df, 
                [bc_to_neighs, bc_to_below_neighs, bc_to_above_neighs],
                tissue_key=tissue_key,
                row_key=row_key,
                col_key=col_key
            )
            #         . . . .
            #        . . . . .
            #       . . . . . .
            #      . . . * . . .
            #       . . . . . *
            #        . . . . .
            #         . . . .
            #
            _add_neighbor(
                curr_bc, 
                c, 
                r, 
                5, 
                -1, 
                row_col_to_barcode, 
                df, 
                [bc_to_neighs, bc_to_below_neighs],
                tissue_key=tissue_key,
                row_key=row_key,
                col_key=col_key
            )
            #         . . . .
            #        . . . . .
            #       . . . . . .
            #      . . . * . . .
            #       . . . . . .
            #        . . . . *
            #         . . . .
            #
            _add_neighbor(
                curr_bc, 
                c, 
                r, 
                4, 
                -2, 
                row_col_to_barcode, 
                df, 
                [bc_to_neighs, bc_to_below_neighs],
                tissue_key=tissue_key,
                row_key=row_key,
                col_key=col_key
            )
            #         . . . .
            #        . . . . .
            #       . . . . . .
            #      . . . * . . .
            #       . . . . . .
            #        . . . . .
            #         . . . *
            #
            _add_neighbor(
                curr_bc, 
                c, 
                r, 
                3, 
                -3, 
                row_col_to_barcode, 
                df, 
                [bc_to_neighs, bc_to_below_neighs],
                tissue_key=tissue_key,
                row_key=row_key,
                col_key=col_key
            )
            #         . . . .
            #        . . . . .
            #       . . . . . .
            #      . . . * . . .
            #       . . . . . .
            #        . . . . .
            #         . . * .
            #
            _add_neighbor(
                curr_bc, 
                c, 
                r, 
                1, 
                -3, 
                row_col_to_barcode, 
                df, 
                [bc_to_neighs, bc_to_below_neighs],
                tissue_key=tissue_key,
                row_key=row_key,
                col_key=col_key
            )
            #         . . . .
            #        . . . . .
            #       . . . . . .
            #      . . . * . . .
            #       . . . . . .
            #        . . . . .
            #         . * . .
            #
            _add_neighbor(
                curr_bc, 
                c, 
                r, 
                -1, 
                -3, 
                row_col_to_barcode, 
                df, 
                [bc_to_neighs, bc_to_below_neighs],
                tissue_key=tissue_key,
                row_key=row_key,
                col_key=col_key
            )
            #         . . . .
            #        . . . . .
            #       . . . . . .
            #      . . . * . . .
            #       . . . . . .
            #        . . . . .
            #         * . . .
            #
            _add_neighbor(
                curr_bc, 
                c, 
                r, 
                -3, 
                -3, 
                row_col_to_barcode, 
                df, 
                [bc_to_neighs, bc_to_below_neighs],
                tissue_key=tissue_key,
                row_key=row_key,
                col_key=col_key
            )
            #         . . . .
            #        . . . . .
            #       . . . . . .
            #      . . . * . . .
            #       . . . . . .
            #        * . . . .
            #         . . . .
            #
            _add_neighbor(
                curr_bc, 
                c, 
                r, 
                -4, 
                -2, 
                row_col_to_barcode, 
                df, 
                [bc_to_neighs, bc_to_below_neighs],
                tissue_key=tissue_key,
                row_key=row_key,
                col_key=col_key
            )
            #         . . . .
            #        . . . . .
            #       . . . . . .
            #      . . . * . . .
            #       * . . . . .
            #        . . . . .
            #         . . . .
            #
            _add_neighbor(
                curr_bc, 
                c, 
                r, 
                -5, 
                -1, 
                row_col_to_barcode, 
                df, 
                [bc_to_neighs, bc_to_below_neighs],
                tissue_key=tissue_key,
                row_key=row_key,
                col_key=col_key
            )
    return bc_to_neighs


def select_regions(adata, regions, cond_key):
    keep_inds = [
        ind
        for ind, ct in zip(adata.obs.index, adata.obs[cond_key])
        if ct in regions
    ]
    adata_copy = adata[keep_inds]
    return adata_copy


def main():
    coords = np.array([
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    ])
    ct_to_inds = {
        'a': [0, 1, 2, 3, 4],
        'b': [5, 6, 7, 8, 9],
        'c': [10, 11, 12, 13, 14]
    }
    new_coords = permute_coords(
        coords,
        ct_to_inds
    )
    print(new_coords)

if __name__ == '__main__':
    main()
