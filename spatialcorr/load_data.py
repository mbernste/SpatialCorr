import pkg_resources as pr
import json
from os.path import join

import anndata

def load_dataset(dataset_id):
    resource_package = __name__

    if dataset_id == 'GSM4284326_P10_ST_rep2':
        data_f = pr.resource_filename(
            resource_package,
            join('datasets', 'GSM4284326_P10_ST_rep2.h5ad')
        )
        adata = anndata.read_h5ad(data_f)
    return adata

