![traval](https://github.com/ArtesiaWater/traval/workflows/traval/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/traval/badge/?version=latest)](https://traval.readthedocs.io/en/latest/?badge=latest)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/d3e9ef5e30724b59a847093daeb6c233)](https://www.codacy.com/gh/ArtesiaWater/traval/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=ArtesiaWater/traval&amp;utm_campaign=Badge_Grade)
[![Codacy Badge](https://app.codacy.com/project/badge/Coverage/d3e9ef5e30724b59a847093daeb6c233)](https://www.codacy.com/gh/ArtesiaWater/traval/dashboard?utm_source=github.com&utm_medium=referral&utm_content=ArtesiaWater/traval&utm_campaign=Badge_Coverage)
![PyPI](https://img.shields.io/pypi/v/traval)

# traval

Tools for applying automatic error detection algorithms to timeseries.

## Introduction

This module is set up to provide tools for applying any error detection 
algorithm to any timeseries. The module consists of three main components:

-   `RuleSet`: the RuleSet object is a highly flexible object for defining error detection algorithms based on (user-defined) functions.
-   `Detector`: a data management object for storing timeseries and error detection results.
-   `SeriesComparison*`: objects for comparing timeseries. These objects include plots for visualizing the comparisons.

The general workflow consists of the following steps:

1.  Define error detection algorithm(s).
2.  Load data, i.e. raw timeseries data and optionally timeseries representing the "truth" to see how well the algorithms perform.
3.  Initialize Detector objects and apply algorithms to timeseries.
4.  Store and analyze the results.

For more detailed information and examples, please refer to the notebooks in 
the examples directory.

## Installation

To install the traval module, follow these steps:

1.  Clone the repository from GitHub.
2.  Open a terminal and navigate to the module root directory: `<your path here>/traval`
3.  Type `pip install -e .`

## Usage

The basic usage of the module is described below. To start using the module, 
import the package:

```python
>>> import traval
```

The first step is generally to define an error detection algorithm. This is 
done with the `RuleSet` object:

```python
>>> ruleset = traval.RuleSet("my_first_algorithm")
```

Add a detection rule (using a general rule from the library contained within 
the module). In this case the rule states any value above 10.0 is suspect:

```python
>>> ruleset.add_rule("rule1", traval.rulelib.rule_ufunc_threshold , apply_to=0, 
                     kwargs={"ufunc": (np.greater,), "threshold": 10.0})
```

Take a look at the ruleset by just typing `ruleset`:

```python
>>> ruleset
```

```text
RuleSet: 'my_first_algorithm'
  step: name            apply_to
     1: rule1                  0
```

Next define a Detector object. This object is designed to store a timeseries 
and the intermediate and final results after applying an error detection 
algorithm. Initialize the Detector object with some timeseries. In this example 
we assume there is a timeseries called `raw_series`:

```python
>>> detect = traval.Detector(raw_series)
```

Apply our first algorithm to the timeseries.

```python
>>> detect.apply_ruleset(ruleset)
```

By default, the result of each step in the algorithm is compared to the 
original series and stored in the `detect.comparisons` attribute. Take a 
look at the comparison between the raw data and the result of the error 
detection algorithm. 

Since we only defined one step, step 1 represents the final result.

```python
>>> cp = detect.comparisons[1]  # result of step 1 = final result
```

The `SeriesComparison*` objects contain methods to visualize the comparison, 
or summarize the number of observations in each category:

```python
>>> cp.plots.plot_series_comparison()  # plot a comparison
>>> cp.summary  # series containing number of observations in each category
```

For more detailed explanation and more complex examples, see the notebook(s) 
in the examples directory.

## Author

-   D.A. Brakenhoff, Artesia, 2020
