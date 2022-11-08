import pandas as pd


class TravalParameters:
    """Object for managing parameters for Traval algorithms.

    Generally not instantiated directly but constructed using one of the
    available classmethods:

    - `TravalParameters.from_ruleset()`
    - `TravalParameters.from_csv()`
    - `TravalParameters.from_json()`
    - `TravalParameters.from_pickle()`

    A logical workflow would be to define a RuleSet, obtain the parameters
    using `TravalParameters.from_ruleset()`, optionally adding a list of
    locations to allow parameters to be set per location. Next, the
    TravalParameters object can be edited and stored using one of the `to_*`
    methods:

    - `TravalParameters.to_csv()`
    - `TravalParameters.to_json()`
    - `TravalParameters.to_pickle()`

    Note that the CSV and JSON options are human-readable but do not really
    support storing callable parameter values. These options are only
    suited to static parameters (i.e. ints, floats and strings). The pickle
    format does support storing callable arguments, but is not human-readable.

    Finally, after selecting one of the storage methods, that file can be
    loaded in a different script to rebuild our RuleSet object. The RuleSet
    object has to be rebuilt explicitly, but the parameters can be obtained
    from the TravalParameters object:

    >>> tp = TravalParameters.from_json("traval_params.json")
    >>> rset = traval.RuleSet("example_algorithm")
    >>> rset.add_rule("gt10",
                      traval.rulelib.rule_ufunc_threshold,
                      apply_to=0,
                      kwargs={
                          "ufunc": (np.greater,),
                          "threshold": tp.get_parameters(rulename="gt10",
                                                         parameter="threshold")
                      })
    """

    def __init__(self, parameters, defaults):
        """Initialize TravalParameters object.

        Parameters
        ----------
        parameters : pandas.DataFrame or None
            DataFrame containing location specific parameters. Index must be
            MultiIndex with named levels: (location, rulename, parameter). If
            None is passed only default parameters are available.
        defaults : pandas.DataFrame
            DataFrame containing default parameters. Index must be
            MultiIndex with named levels: (location, rulename, parameter).
        """
        self.parameters = self._check_params_dataframe(parameters)
        self.defaults = self._check_params_dataframe(defaults)

    def __repr__(self):
        """Representation of TravalParameters object."""
        return self._combine_parameter_dfs().__repr__()

    @property
    def n_locations(self):
        """Number of locations for which parameters are stored."""
        return len(self.locations)

    @property
    def locations(self):
        """List of locations."""
        if self.parameters is not None:
            return self.parameters.index.get_level_values(0).unique().tolist()
        else:
            return []

    @property
    def rulenames(self):
        """List of unique rule names."""
        df = self._combine_parameter_dfs()
        if not df.empty:
            return df.index.get_level_values(1).unique().tolist()
        else:
            return []

    @staticmethod
    def _split_df(params):
        """Split DataFrame into default and location-specific parameters.

        Parameters
        ----------
        params : pandas.DataFrame
            DataFrame containing all parameters

        Returns
        -------
        parameters : pandas.DataFrame
            DataFrame containing location specific parameters
        defaults : pandas.DataFrame
            DataFrame containing default parameters
        """
        mask = params.index.get_level_values(0) == "default"
        idx = pd.IndexSlice
        defaults = params.loc[idx[mask, :, :], :]
        parameters = params.loc[idx[~mask, :, :], :]
        return parameters, defaults

    @staticmethod
    def _check_params_dataframe(df):
        """Check type and/or if DataFrame index has correct level names.

        Parameters
        ----------
        df : object
            object to check

        Returns
        -------
        df : None or pandas.DataFrame
            returns object if checks pass

        Raises
        ------
        ValueError
            if object type not understood, or index names are incorrect.
        """
        if df is None:
            return df
        elif isinstance(df, pd.DataFrame):
            idxnames = {"location", "rulename", "parameter"}
            missing = idxnames.difference(df.index.names)
            if len(missing) > 0:
                raise ValueError("Parameter DataFrame index does not contain"
                                 f"required levels/names. Expected {idxnames},"
                                 f" got: {df.index.names}")
            return df
        else:
            return ValueError("Parameter DataFrame type not understood: "
                              f"{type(df)}")

    @classmethod
    def from_csv(cls, csvfile):
        """Create TravalParameters object from CSV-file.

        Note: parameter value dtypes are preserved for common dtypes when
        writing to and reading data from CSV files. Callable parameters
        are converted to string after writing to CSV.

        Parameters
        ----------
        csvfile : str
            path to CSV-file

        Returns
        -------
        TravalParameters
        """
        params = pd.read_csv(csvfile, index_col=[0, 1, 2])
        params.sort_index(inplace=True)
        for i, (v, t) in params.loc[:, ["value", "dtype"]].iterrows():
            if t == "float":
                v = float(v)
            elif t == "int":
                v = int(v)
            elif t == "str":
                continue  # already str
            elif t == "NoneType":
                v = None
            params.loc[i, "value"] = v
        params.drop(columns=['dtype'], inplace=True)
        parameters, defaults = cls._split_df(params)
        return cls(parameters, defaults)

    @classmethod
    def from_json(cls, jsonfile):
        """Create TravalParameters object from JSON-file.

        Note: parameter value dtypes are preserved for common dtypes.
        Callable parameters are unrecognizable after writing to and reading
        from JSON.

        Parameters
        ----------
        jsonfile : str
            path to JSON-file

        Returns
        -------
        TravalParameters
        """
        params = pd.read_json(jsonfile, orient="table", typ="frame")
        parameters, defaults = cls._split_df(params)
        return cls(parameters, defaults)

    @classmethod
    def from_pickle(cls, fname):
        """Create TravalParameters object from pickle.

        Note: parameter value dtypes are preserved when writing to and
        reading from pickle files. Pickle files are not human-readable.

        Parameters
        ----------
        fname : str
            path to pickle file

        Returns
        -------
        TravalParameters
        """
        import pickle
        with open(fname, "rb") as f:
            params = pickle.load(f)
        parameters, defaults = cls._split_df(params)
        return cls(parameters, defaults)

    @classmethod
    def from_ruleset(cls, rset, locations=None):
        """Create TravalParameters object from RuleSet object.

        Parameters
        ----------
        rset : traval.RuleSet
            RuleSet object containing error-detection algorithm
        locations : list of str, optional
            list of locations, by default None. Passing this list copies
            the current RuleSet for each passed location in the list.

        Returns
        -------
        TravalParameters

        Raises
        ------
        ValueError
            if locations kwarg is not understood
        """
        params = rset.get_parameters()
        if locations is None:
            df = None
        elif isinstance(locations, list):
            collect = []
            for iloc in locations:
                pi = params.copy()
                pi["location"] = iloc
                collect.append(pi)
            df = pd.concat(collect, axis=0)
            df = df.set_index(["location", "rulename", "parameter"])
            df = df.loc[:, ["value"]]
        else:
            raise ValueError("locations must be a list of str!")
        params["location"] = "default"
        params = params.set_index(["location", "rulename", "parameter"])
        params = params.loc[:, ["value"]]
        return cls(df, params)

    def get_parameters(self, rulename=None, location=None,
                       parameter=None, squeeze=True):
        """Get parameter(s) by querying DataFrame.

        Parameters
        ----------
        rulename : str, optional
            name of rule to get parameter values for, by default None
        location : str, optional
            name of location to get parameter values for, by default None,
            in which case the default parameters are used.
        parameter : str, optional
            name of parameter to get values for, by default None
        squeeze : bool, optional
            If query results in a single parameter value, return the
            value and not the DataFrame, by default True

        Returns
        -------
        p : pandas.DataFrame or Any
            returns single parameter value if possible and squeeze=True.
            Otherwise returns DataFrame if query results in multiple
            parameters.

        Raises
        ------
        KeyError
            if location is not in parameters DataFrame
        ValueError
            if there are no location specific parameters
        """
        idx = pd.IndexSlice
        if location is None:
            p = self.defaults
        elif self.parameters is not None:
            mask = self.parameters.index.get_level_values(0) == location
            if mask.sum() == 0:
                raise KeyError(f"Location '{location}' not in parameters"
                               " DataFrame!")
            p = self.parameters.loc[idx[mask, :, :], :]
        else:
            raise ValueError("No location specific parameters!")

        indexer = (
            slice(None),
            rulename if rulename is not None else slice(rulename),
            parameter if parameter is not None else slice(parameter),
        )
        if squeeze:
            return p.loc[indexer, "value"].squeeze()
        else:
            return p.loc[indexer, "value"]

    def get_parameters_as_dict(self, rulename, location=None):
        """Get all parameters for one rule as dictionary.

        Parameters
        ----------
        rulename : str
            name of rule to get parameter values for
        location : str, optional
            location to get parameter values for, by default None, in which
            case the default parameters are used.

        Returns
        -------
        p : dict
            dictionary containing parameter names as keys van values as values
        """
        p = self.get_parameters(rulename=rulename, location=location)
        return p.droplevel([0, 1], axis=0).to_dict()

    def update_parameter_value(self, location, rulename, parameter, value):
        """Update location specific parameter value.

        Parameters
        ----------
        location : str
            location to update parameter for
        rulename : str
            name of rule
        parameter : str
            name of parameter
        value : Any
            value of the parameter
        """
        if self.parameters is None:
            raise ValueError("No location specific parameters!")
        self.parameters.loc[(location, rulename, parameter)] = value

    def update_default_value(self, rulename, parameter, value):
        """Update default parameter value.

        Parameters
        ----------
        rulename : str
            name of rule
        parameter : str
            name of parameter
        value : Any
            value of the parameter
        """
        self.defaults.loc[("default", rulename, parameter)] = value

    def delete_parameter_value(self, location, rulename, parameter):
        """Delete location specific parameter value.

        Parameters
        ----------
        location : str
            location to update parameter for
        rulename : str
            name of rule
        parameter : str
            name of parameter
        """
        if self.parameters is None:
            raise ValueError("No location specific parameters!")
        self.parameters.drop(index=(location, rulename, parameter),
                             inplace=True)

    def delete_default_value(self, rulename, parameter):
        """Delete default parameter value.

        Parameters
        ----------
        rulename : str
            name of rule
        parameter : str
            name of parameter
        """
        self.parameters.drop(index=("default", rulename, parameter),
                             inplace=True)

    def _combine_parameter_dfs(self):
        """Concatenate default and location specific parameter DataFrames.

        Returns
        -------
        df : pandas.DataFrame
            Full DataFrame containing both default and location specific
            parameters.
        """
        if self.parameters is None:
            p = pd.DataFrame()
        else:
            p = self.parameters
        df = pd.concat([self.defaults, p], axis=0)
        return df

    @staticmethod
    def _test_callable(f):
        """Method to test whether parameter value is a callable.

        Also returns True if callable is stored in a tuple.

        Parameters
        ----------
        f : Any
            parameter value

        Returns
        -------
        bool
            True if parameter value is callable or tuple containing a
            callable, otherwise returns False
        """
        if isinstance(f, tuple):
            return callable(f[0])
        else:
            return callable(f)

    def to_csv(self, csvfile, only_static_params=True):
        """Write TravalParameters to CSV-file.

        Parameters
        ----------
        csvfile : str
            path to CSV-files
        only_static_params : bool, optional
            export non-callable parameters only, by default True. Callable
            parameters are converted to string when writing to CSV.
        """
        df = self._combine_parameter_dfs()
        df["dtype"] = df["value"].apply(lambda o: type(o).__name__)
        if only_static_params:
            mask = df["value"].apply(self._test_callable)
            df.loc[~mask].to_csv(csvfile)
        else:
            df.to_csv(csvfile)

    def to_json(self, jsonfile, only_static_params=True):
        """Write TravalParameters to JSON-file.

        Parameters
        ----------
        jsonfile : str
            path to JSON-file
        only_static_params : bool, optional
            export non-callable parameters only, by default True. Callable
            parameters are not preserved  and become meaningless
            when writing to JSON.
        """
        df = self._combine_parameter_dfs()
        if only_static_params:
            mask = df["value"].apply(self._test_callable)
            df = df.loc[~mask]
        df.to_json(jsonfile, orient="table")

    def to_pickle(self, fname):
        """Write TravalParameters to pickle.

        Parameters
        ----------
        fname : str
            path to pickle file.
        """
        df = self._combine_parameter_dfs()
        df.to_pickle(fname)
