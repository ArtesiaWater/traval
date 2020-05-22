from collections import OrderedDict
import pandas as pd
import numpy as np


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
        """Apply ruleset to series

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
        """add rule to RuleSet

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
        """delete rule from RuleSet

        Parameters
        ----------
        name : str
            name of the rule to delete

        """
        self.rules.pop(name)
        # logger.debug(f"Removed {name} from ruleset!")

    def update_rule(self, name, func, apply_to=None, kwargs=None):
        """Update rule in RuleSet

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
        """convert RuleSet to pandas.DataFrame

        Returns
        -------
        rdf : pandas.DataFrame
            DataFrame containing all the information from the RuleSet

        """
        rules = [irule for irule in self.rules.values()]
        rdf = pd.DataFrame(rules, index=range(1, len(rules) + 1))
        rdf.index.name = "step"
        return rdf

    def get_parameters(self):
        pass

    def _parse_kwargs(self, kwargs, name=None):
        """internal method, parse keyword arguments dictionary

        Iterates over keys, values in kwargs dictionary. If value is callable,
        calls value with 'name' as function argument. The result is stored
        in a new dictionary with the original key.

        Parameters
        ----------
        kwargs: dict
            dictionary of arguments
        name: str, optional
            function argument for callable kwargs(usually a series name)

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
        """internal method, apply ruleset to series

        Parameters
        ----------
        series: pandas.Series or pandas.DataFrame
            timeseries to apply rules to

        Returns
        -------
        d: OrderedDict
            Dictionary containing resulting timeseries after applying rules.
            Keys represent step numbers(0 is the original series, 1 the
            outcome of rule  # 1, etc.)
        c: OrderedDict
            Dictionary containing corrections to timeseries based on rules
            Keys represent step numbers(1 contains the corrections based on
            rule  # 1, etc.). When no correction is available, step contains
            the value 0.

        """
        name = series.name
        d, c = OrderedDict(), OrderedDict()  # store results, corrections
        d[0] = series
        for i, irule in enumerate(self.rules.values(), start=1):
            # if apply_to is int, apply to that series
            if isinstance(irule["apply_to"], int):
                # parse dict, if callable call func and use result as kwarg
                arg_dict = self._parse_kwargs(irule["kwargs"], name)
                corr = irule["func"](d[int(irule["apply_to"])], **arg_dict)
                # store both correction and result
                d[i] = d[int(irule["apply_to"])] + corr
                c[i] = corr
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
                c[i] = pd.Series(index=d[i].index,
                                 data=np.zeros(d[i].index.size),
                                 dtype=float,
                                 fastpath=True)
            else:
                raise TypeError("Value of 'apply_to' must be int or tuple "
                                f"of ints. Got '{irule['apply_to']}'")
        return d, c


if __name__ == "__main__":

    name1 = "gt10"

    def func1(s):
        mask = s > 10
        s = pd.Series(index=s.index, data=0.0)
        s.loc[mask] = np.nan
        return s

    name2 = "lt0_very_long_name"

    def func2(s):
        mask = s < 0
        s = pd.Series(index=s.index, data=0.0)
        s.loc[mask] = np.nan
        return s

    rset = RuleSet("my_rulez")
    rset.add_rule(name1, func1, apply_to=0, kwargs=None)
    rset.add_rule(name2, func2, apply_to=1, kwargs=None)

    rdf = rset.to_dataframe()

    series = pd.Series(index=range(10), data=range(-5, 23, 3))

    d, c = rset(series)
