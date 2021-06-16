.. traval documentation master file, created by
   sphinx-quickstart on Wed Jun 16 11:27:20 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to traval's documentation!
==================================

Python package for applying automatic error detection algorithms to timeseries.

This module is set up to provide tools for applying any error detection 
algorithm to any timeseries. The module consists of three main components:

-   `Detector`: a data management object for storing timeseries and error detection results.
-   `RuleSet`: the RuleSet object is a highly flexible object for defining error detection algorithms based on (user-defined) functions.
-   `SeriesComparison*`: objects for comparing timeseries. These objects include plots for visualizing the comparisons.

The general workflow consists of the following steps:

1.  Define error detection algorithm(s).
2.  Load data, i.e. raw timeseries data and optionally timeseries representing the "truth" to see how well the algorithms perform.
3.  Initialize Detector objects and apply algorithms to timeseries.
4.  Store and analyze the results.

For more detailed information and examples, please refer to the notebooks in 
the examples directory.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   Getting Started <getting_started>
   Examples <examples>
   API-docs <modules>

Indices and tables
==================

* :ref:`genindex`
