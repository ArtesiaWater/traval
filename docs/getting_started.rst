Getting Started
===============


Installation
------------

To install traval, a working version of Python 3.7 or 3.8 has to be installed on 
your computer. We recommend using the Anaconda Distribution with Python 3.7 as 
it includes most of the python package dependencies and the Jupyter Notebook 
software to run the notebooks. However, you are free to install any 
Python distribution you want. 

To install traval, use:

.. code:: bash

    pip install traval

To install in development mode, clone the repository, then type the following 
from the module root directory:

.. code:: bash

    pip install -e .


Usage
-----

The basic usage of the module is described below. To start using the module, 
import the package:

.. code:: python

    import metran


The first step is generally to define an error detection algorithm. This is 
done with the `RuleSet` object:

.. code:: python

    ruleset = traval.RuleSet("my_first_algorithm")


Add a detection rule (using a general rule from the library contained within 
the module). In this case the rule states any value above 10.0 is suspect:

.. code-block:: python

    ruleset.add_rule("rule1", 
                     traval.rulelib.rule_ufunc_threshold , 
                     apply_to=0, 
                     kwargs={"ufunc": (np.greater,), "threshold": 10.0}
                     )


Take a look at the ruleset by just typing `ruleset`:


.. code:: python
    
    ruleset


.. code-block::
    
    RuleSet: 'my_first_algorithm'
       step: name            apply_to
          1: rule1                  0


Next define a Detector object. This object is designed to store a timeseries 
and the intermediate and final results after applying an error detection 
algorithm. Initialize the Detector object with some timeseries. In this example 
we assume there is a timeseries called `raw_series`:


.. code:: python
    
    detect = traval.Detector(raw_series)


Apply our first algorithm to the timeseries.

.. code:: python

    detect.apply_ruleset(ruleset)


By default, the result of each step in the algorithm is compared to the 
original series and stored in the `detect.comparisons` attribute. Take a 
look at the comparison between the raw data and the result of the error 
detection algorithm. 

Since we only defined one step, step 1 represents the final result.

.. code:: python

    cp = detect.comparisons[1]  # result of step 1 = final result


The `SeriesComparison*` objects contain methods to visualize the comparison, 
or summarize the number of observations in each category:

.. code-block:: python

    cp.plots.plot_series_comparison()  # plot a comparison
    cp.summary  # series containing number of observations in each category


For more detailed explanation and more complex examples, see the notebook(s) 
in the examples directory.
