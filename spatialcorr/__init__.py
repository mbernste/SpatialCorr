from .statistical_test import run_test, run_test_between_region_pairs, est_corr_cis, covariance_kernel_estimation, compute_kernel_matrix
from .plot import plot_slide, plot_filtered_spots, plot_correlation, plot_local_scatter
from .utils import permute_coords, map_row_col_to_barcode, compute_neighbors, most_significant_pairs, compute_local_correlation
from .load_data import load_dataset
from .wrappers import analysis_pipeline_pair, analysis_pipeline_set, kernel_diagnostics

__version__ = "1.2.0"
