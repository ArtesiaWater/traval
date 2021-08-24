import json
import warnings
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import pandas as pd

from . import rulelib


class RuleSetEncoder(json.JSONEncoder):
    def default(self, o):
        if callable(o):
            return "func:" + o.__name__
        elif isinstance(o, pd.Series):
            return "series:" + o.to_json(date_format="iso", orient="split")
        elif isinstance(o, pd.DataFrame):
            # Necessary to maintain order when using the JSON format!
            return "dataframe:" + o.to_json(orient="index")
        elif pd.isna(o):
            return None
        else:
            return super(RuleSetEncoder, self).default(o)


def ruleset_hook(obj):

    for key, value in obj.items():
        if str(value).startswith("func:"):
            # from rlib
            funcname = value.split(":")[1]
            try:
                val = getattr(rulelib, funcname)
            except AttributeError:
                warnings.warn(f"Could not load function {funcname} "
                              "from `traval.rulelib`!")
                val = funcname
            obj[key] = val
        elif key in ['ufunc']:
            # numpy functions
            funcname = value[0].split(":")[1]
            try:
                val = getattr(np, funcname)
            except AttributeError:
                warnings.warn(f"Could not load function {funcname} "
                              "from `numpy`!")
                val = (funcname,)
            obj[key] = (val,)
        elif str(value).startswith("series:"):
            try:
                value = value[7:]  # strip 'series:'
                obj[key] = pd.read_json(value, typ='series', orient="split")
            except Exception:
                obj[key] = value
            if isinstance(obj[key], pd.Series):
                obj[key].index = obj[key].index.tz_localize(None)
        elif str(value).startswith("dataframe:"):
            # Necessary to maintain order when using the JSON format!
            value = value[9:]  # strip 'dataframe:'
            value = json.loads(value, object_pairs_hook=OrderedDict)
            df = pd.DataFrame(data=value, columns=value.keys()).T
            obj[key] = df.apply(pd.to_numeric, errors="ignore")
        else:
            try:
                obj[key] = json.loads(value, object_hook=ruleset_hook)
            except Exception:
                obj[key] = value
    return obj


class RuleSet:
    """Create RuleSet object for storing detection rules.

    The RuleSet object stores detection rules and other relevant information
    in a dictionary. The order in which rules are carried out, the functions
    that parse the timeseries, the extra arguments required by those functions
    are all stored together.

    The detection functions must take a series as the first argument, and
    return a series with corrections based on the detection rule. In the
    corrections series invalid values are set to np.nan, and adjustments are
    defined with a float. No change is defined as 0. Extra keyword arguments
    for the function can be passed through a kwargs dictionary. These kwargs
    are also allowed to contain functions. These functions must return some
    value based on the name of the series.


    Parameters
    ----------
    name : str, optional
        name of the RuleSet, by default None


    Examples
    --------

    Given two detection functions 'foo' and 'bar':

    >>> rset = RuleSet(name="foobar")
    >>> rset.add_rule("foo", foo, apply_to=0)  # add rule 1
    >>> rset.add_rule("bar", bar, apply_to=1, kwargs={"n": 2})  # add rule 2
    >>> print(rset)  # print overview of rules
    """

    def __init__(self, name=None):
        """Create RuleSet object for storing detection rules.

        Parameters
        ----------
        name : str, optional
            name of the RuleSet, by default None
        """
        self.rules = OrderedDict()
        self.name = name if name is not None else ""

    def __repr__(self):
        """String representation of object."""
        description = f"RuleSet: '{self.name}'"
        header = "  {step:>4}: {name:<15} {apply_to:<8}".format(
            step="step", name="name", apply_to="apply_to")
        rows = []
        tmplt = "  {step:>4g}: {name:<15} {apply_to:>8}"
        for i, (inam, irow) in enumerate(self.rules.items()):
            rows.append(tmplt.format(step=i + 1, name=inam[:15],
                                     apply_to=str(irow["apply_to"])))

        return "\n".join([description, header] + rows)

    def __call__(self, series):
        """Apply ruleset to series.

        Parameters
        ----------
        series : pandas.Series or pandas.DataFrame
            timeseries to apply rules to

        Returns
        -------
        d : OrderedDict
            Dictionary containing resulting timeseries after applying rules.
            Keys represent step numbers (0 is the original series, 1 the
            outcome of rule #1, etc.)
        c : OrderedDict
            Dictionary containing corrections to timeseries based on rules
            Keys represent step numbers (1 contains the corrections based on
            rule #1, etc.). When no correction is available, step contains
            the value 0.
        """

        return self._applyself(series)

    def add_rule(self, name, func, apply_to=None, kwargs=None):
        """Add rule to RuleSet.

        Parameters
        ----------
        name : str
            name of the rule
        func : callable
            function that takes series as input and returns
            a correction series.
        apply_to : int or tuple of ints, optional
            series to apply the rule to, by default None, which defaults to the
            original series. E.g. 0 is the original series, 1 is the result of
            step 1, etc. If a tuple of ints is passed, the results of those
            steps are collected and passed to func.
        kwargs : dict, optional
            dictionary of additional keyword arguments for func, by default
            None. Additional arguments can be functions as well, in which case
            they must return some value based on the name of the series to
            which the RuleSet will be applied.
        """
        self.rules[name] = {"name": name, "func": func,
                            "apply_to": apply_to, "kwargs": kwargs}

    def del_rule(self, name):
        """Delete rule from RuleSet.

        Parameters
        ----------
        name : str
            name of the rule to delete
        """
        self.rules.pop(name)
        # logger.debug(f"Removed {name} from ruleset!")

    def update_rule(self, name, func, apply_to=None, kwargs=None):
        """Update rule in RuleSet.

        Parameters
        ----------
        name : str
            name of the rule
        func : callable
            function that takes series as input and returns
            a correction series.
        apply_to : int or tuple of ints, optional
            series to apply the rule to, by default None, which defaults to the
            original series. E.g. 0 is the original series, 1 is the result of
            step 1, etc. If a tuple of ints is passed, the results of those
            steps are collected and passed to func.
        kwargs : dict, optional
            dictionary of additional keyword arguments for func, by default
            None. Additional arguments can be functions as well, in which case
            they must return some value based on the name of the series to
            which the RuleSet will be applied.
        """
        if name not in self.rules.keys():
            raise KeyError("No rule by that name in RuleSet!")
        self.rules.update({name: {"name": name, "func": func,
                                  "apply_to": apply_to, "kwargs": kwargs}})

    def get_step_name(self, istep):
        if istep > 0:
            n = list(self.rules.keys())[istep - 1]
        elif istep == 0:
            n = "base series"
        else:
            # negative step counts from end
            n = list(self.rules.keys())[istep]
        return n

    def to_dataframe(self):
        """Convert RuleSet to pandas.DataFrame.

        Returns
        -------
        rdf : pandas.DataFrame
            DataFrame containing all the information from the RuleSet
        """
        rules = self.rules.values()
        rdf = pd.DataFrame(rules, index=range(1, len(rules) + 1))
        rdf.index.name = "step"
        return rdf

    def get_parameters(self):
        cols = ["rulename", "step", "func", "parameter", "value"]
        params = pd.DataFrame(columns=cols)
        counter = 0
        for rnam, irule in self.rules.items():
            if irule["kwargs"] is None:
                continue
            for name, value in irule["kwargs"].items():
                params.loc[counter, cols] = \
                    rnam, irule["apply_to"], irule["func"], name, value
                counter += 1
        return params

    @staticmethod
    def _parse_kwargs(kwargs, name=None):
        """Internal method, parse keyword arguments dictionary.

        Iterates over keys, values in kwargs dictionary. If value is callable,
        calls value with 'name' as function argument. The result is stored
        in a new dictionary with the original key.

        Parameters
        ----------
        kwargs: dict
            dictionary of arguments
        name: str, optional
            function argument for callable kwargs (usually a series name)

        Returns
        -------
        dict
            dictionary of parsed arguments
        """
        new_args = dict()
        if kwargs is not None:
            for k, v in kwargs.items():
                if callable(v):
                    new_args[k] = v(name)
                else:
                    new_args[k] = v
        return new_args

    def _applyself(self, series):
        """Internal method, apply ruleset to series.

        Parameters
        ----------
        series: pandas.Series or pandas.DataFrame
            timeseries to apply rules to

        Returns
        -------
        d: OrderedDict
            Dictionary containing resulting timeseries after applying rules.
            Keys represent step numbers (0 is the original series, 1 the
            outcome of rule  # 1, etc.)
        c: OrderedDict
            Dictionary containing corrections to timeseries based on rules
            Keys represent step numbers(1 contains the corrections based on
            rule  # 1, etc.). When no correction is available, step contains
            the value 0.
        """
        name = series.name
        d, c = {}, {}  # store results, corrections
        d[0] = series
        for i, irule in enumerate(self.rules.values(), start=1):
            # if apply_to is int, apply to that series
            if isinstance(irule["apply_to"], int):
                # parse dict, if callable call func and use result as kwarg
                arg_dict = self._parse_kwargs(irule["kwargs"], name)
                corr = irule["func"](d[int(irule["apply_to"])], **arg_dict)
                # store both correction and result
                d[i] = d[int(irule["apply_to"])] + corr
                c[i] = corr.loc[corr != 0.0].copy()
            # if apply_to is tuple, collect series as kwargs to func
            elif isinstance(irule["apply_to"], tuple):
                # collect results
                collect_args = []
                for n in irule["apply_to"]:
                    collect_args.append(d[n])
                # parse dict, if callable call func and use result as kwarg
                arg_dict = self._parse_kwargs(irule["kwargs"], name)
                # apply func with collected results
                # store both correction and result
                d[i] = irule["func"](*collect_args, **arg_dict)
                c[i] = np.zeros(1)
            else:
                raise TypeError("Value of 'apply_to' must be int or tuple "
                                f"of ints. Got '{irule['apply_to']}'")
        return d, c

    def get_rule(self, istep=None, stepname=None):
        if istep is not None:
            istepname = self.get_step_name(istep)
            irule = self.rules[istepname]
        elif stepname is not None:
            irule = self.rules[stepname]
        else:
            raise ValueError("Provide one of 'istep' or 'stepname'!")
        return irule

    def get_func(self, istep=None, stepname=None):
        irule = self.get_rule(istep=istep, stepname=stepname)
        return irule["func"]

    def get_applyto(self, istep=None, stepname=None):
        irule = self.get_rule(istep=istep, stepname=stepname)
        return irule["applyto"]

    def get_kwargs(self, istep=None, stepname=None, kwarg_name=None):
        irule = self.get_rule(istep=istep, stepname=stepname)
        arg_dict = self._parse_kwargs(irule["kwargs"], name=kwarg_name)
        return arg_dict

    def to_pickle(self, fname, verbose=True):
        """Write RuleSet to disk as pickle.

        Parameters
        ----------
        fname : str
            filename or path of file
        verbose : bool, optional
            prints message when operation complete, default is True

        See also
        --------
        from_pickle : load RuleSet from pickle file
        to_json : store RuleSet as json file (does not support custom functions)
        from_json : load RuleSet from json file
        """
        import pickle
        rules = deepcopy(self.rules)
        rules["name"] = self.name
        with open(fname, "wb") as f:
            pickle.dump(rules, f)
        if verbose:
            print(f"RuleSet written to file: '{fname}'")

    @classmethod
    def from_pickle(cls, fname):
        """Load RuleSet object form pickle file.

        Parameters
        ----------
        fname : str
            filename or path to file

        Returns
        -------
        RuleSet
            RuleSet object, including custom functions and parameters

        See also
        --------
        to_pickle : store RuleSet as pickle (supports custom functions)
        to_json : store RuleSet as json file (does not support custom functions)
        from_json : load RuleSet from json file
        """
        import pickle
        with open(fname, "rb") as f:
            rules = pickle.load(f)
        rs = cls(name=rules.pop("name"))
        rs.rules.update(rules)
        return rs

    def to_json(self, fname, verbose=True):
        """Write RuleSet to disk as json file.

        Note that it is not possible to write custom functions to a JSON
        file. When writing the JSON only the name of the function is stored.
        When loading a JSON file, the function name is used to search within
        `traval.rulelib`. If the function can be found, it loads that
        function. A RuleSet making use of functions in the default rulelib.

        Parameters
        ----------
        fname : str
            filename or path to file
        verbose : bool, optional
            prints message when operation complete, default is True


        See also
        --------
        from_json : load RuleSet from json file
        to_pickle : store RuleSet as pickle (supports custom functions)
        from_pickle : load RuleSet from pickle file
        """
        msg = ("Custom functions will not be preserved when storing "
               "RuleSet as JSON file!")
        warnings.warn(msg)
        rules = deepcopy(self.rules)
        rules["name"] = self.name
        if not fname.endswith(".json"):
            raise ValueError("Filename requires '.json' as extension!")
        with open(fname, "w") as f:
            json.dump(rules, f, indent=4, cls=RuleSetEncoder)
        if verbose:
            print(f"RuleSet written to file: '{fname}'")

    @classmethod
    def from_json(cls, fname):
        """Load RuleSet object from JSON file.

        Attempts to load functions in the RuleSet by searching for the
        function name in traval.rulelib. If the function cannot be found, only
        the name of the function is preserved. This means a RuleSet
        with custom functions will not be fully functional when loaded
        from a JSON file.

        Parameters
        ----------
        fname : str
            filename or path to file

        Returns
        -------
        RuleSet:
            RuleSet object

        See also
        --------
        to_json : store RuleSet as JSON file (does not support custom functions)
        to_pickle : store RuleSet as pickle (supports custom functions)
        from_pickle : load RuleSet from pickle file
        """
        with open(fname, "r") as f:
            data = json.load(f, object_hook=ruleset_hook)

        name = data.pop("name")
        rset = cls(name=name)
        for k, v in data.items():
            rset.add_rule(k, v['func'], apply_to=v['apply_to'],
                          kwargs=v["kwargs"])
        return rset
