API
===

Pre-built pipelines
-------------------

The following wrapper function creates plots to diagnose the spatial kernel used in SpatialCorr's statistical tests.

.. autofunction:: spatialcorr.kernel_diagnostics

It outputs the following multi-panel figure:

.. raw:: html

    <p align="center">
    <img src="https://raw.githubusercontent.com/mbernste/spatialcorr/main/doc/_static/img/kernel_diagnostic.png"/>
    </p>

The following wrapper function implements a full analysis pipeline for investigating spatially varying correlation between a pair of genes.

.. autofunction:: spatialcorr.analysis_pipeline_pair 

It outputs the following multi-panel figure:

.. raw:: html

    <p align="center">
    <img src="https://raw.githubusercontent.com/mbernste/spatialcorr/main/doc/_static/img/KRT16_KRT6B_analysis.png"/>
    </p>

The following wrapper function implements a full analysis pipeline for investigating spatially varying correlation between a set of genes.

.. autofunction:: spatialcorr.analysis_pipeline_set

It outputs the following multi-panel figure:

.. raw:: html

    <p align="center">
    <img src="https://raw.githubusercontent.com/mbernste/spatialcorr/main/doc/_static/img/keratin_gene_set_analysis.png"/>
    </p>


Statistical
-----------

.. autofunction:: spatialcorr.run_test

.. autofunction:: spatialcorr.run_test_between_region_pairs

.. autofunction:: spatialcorr.est_corr_cis


Plotting
--------

.. autofunction:: spatialcorr.plot.plot_correlation

.. raw:: html

    <p align="center">
    <img src="https://raw.githubusercontent.com/mbernste/spatialcorr/main/doc/_static/img/KRT16_KRT6B_correlation.png"/>
    </p>

.. autofunction:: spatialcorr.plot.plot_ci_overlap

.. raw:: html

    <p align="center">
    <img src="https://raw.githubusercontent.com/mbernste/spatialcorr/main/doc/_static/img/KRT16_KRT6B_correlation_ci.png"/>
    </p>

.. autofunction:: spatialcorr.plot.plot_local_scatter

.. raw:: html

    <p align="center">
    <img src="https://raw.githubusercontent.com/mbernste/spatialcorr/main/doc/_static/img/KRT16_KRT6B_local_correlation_scatter.png"/>
    </p>

.. autofunction:: spatialcorr.plot.region_scatterplots

.. raw:: html

    <p align="center">
    <img src="https://raw.githubusercontent.com/mbernste/spatialcorr/main/doc/_static/img/KRT17_KRT6B_region_scatters.png"/>
    </p>

.. autofunction:: spatialcorr.plot.mult_genes_plot_correlation

.. raw:: html

    <p align="center">
    <img src="https://raw.githubusercontent.com/mbernste/spatialcorr/main/doc/_static/img/keratin_gene_corr_heatmaps.png"/>
    </p>

.. autofunction:: spatialcorr.plot.cluster_pairwise_correlations

.. raw:: html

    <p align="center">
    <img src="https://raw.githubusercontent.com/mbernste/spatialcorr/main/doc/_static/img/keratin_gene_corr_dendrogram.png"/>
    </p>

.. autofunction:: spatialcorr.plot.plot_filtered_spots

.. raw:: html

    <p align="center">
    <img src="https://raw.githubusercontent.com/mbernste/spatialcorr/main/doc/_static/img/filt_spots.png"/>
    </p>

.. autofunction:: spatialcorr.plot.plot_slide

Helper functions
-----------------

.. autofunction:: spatialcorr.compute_local_correlation

.. autofunction:: spatialcorr.most_significant_pairs

.. autofunction:: spatialcorr.compute_kernel_matrix

.. autofunction:: spatialcorr.covariance_kernel_estimation

Datasets
--------

.. autofunction:: spatialcorr.load_dataset
