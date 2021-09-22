API
===

Wrapper functions
-----------------

The following wrapper function creates plots to diagnose the spatial kernel used in SpatialDC's statistical tests.

.. autofunction:: kernel_diagnostics

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


Statistical tests
-----------------

.. autofunction:: spatialcorr.run_test
