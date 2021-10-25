import pkg_resources as pr
import json
from os.path import join

import anndata

def load_dataset(dataset_id):
    """Load a prepackaged spatial gene expression dataset.

    Parameters
    ----------
    dataset_id : string, Options: {'GSM4284326_P10_ST_rep2'}
        The ID of the dataset to load.

    Returns
    -------
    adata : AnnData
        The spatial gene expression dataset. The rows and column
        coordinates are stored in `adata.obs['row']` and `adata.obs['col']`
        respectively. The clusters are stored in `adata.obs['cluster']`.
        The gene expression matrix `adata.X` is in units of Dino normalized
        expression values. 
    """
    resource_package = __name__

    if dataset_id == 'GSM4284326_P10_ST_rep2':
        data_f = pr.resource_filename(
            resource_package,
            join('datasets', 'GSM4284326_P10_ST_rep2.h5ad')
        )
        adata = anndata.read_h5ad(data_f)
    return adata

