from collections import defaultdict

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
            if df.loc[bc_cand][tissue_key] == 1:
                for d in dicts:
                    d[curr_bc].append(bc_cand)
        except KeyError:
            pass


def map_coords_to_neighbors(
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
    df_ts = df.loc[df[tissue_key] == 1]
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
    return bc_to_neighs, bc_to_above_neighs, bc_to_below_neighs

