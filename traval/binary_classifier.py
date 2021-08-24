import numpy as np
import pandas as pd


class BinaryClassifier:
    """Class for calculating binary classification statistics."""

    stats_abbreviations = {
        "tp": "true positives",
        "fp": "false positives",
        "fn": "false negatives",
        "tn": "true negatives",
        "sensitivity": "sensitivity",
        "tpr": "true positive rate",
        "fnr": "false negative rate",
        "specificity": "specificity",
        "tnr": "true negative rate",
        "fpr": "false positive rate",
        "ppv": "positive predictive value",
        "npv": "negative predictive value",
        "fdr": "false discovery rate",
        "for": "false omission rate",
        "acc": "accuracy",
        "prev": "prevalence",
        "informedness": "informedness",
        "mcc": "matthews correlation coefficient",
    }

    def __init__(self, tp, fp, tn, fn):
        """Initialize class for calculating binary classification statistics.

        Parameters
        ----------
        tp : int
            number of True Positives (TP)
        fp : int
            number of False Positives (FP)
        tn : int
            number of True Negatives (TN)
        fn : int
            number of False Negatives (FN)
        """
        self.n_obs = tp + fp + tn + fn
        self.true_positives = self.tp = tp
        self.false_positives = self.fp = fp
        self.true_negatives = self.tn = tn
        self.false_negatives = self.fn = fn

    @classmethod
    def from_series_comparison_relative(cls, comparison):
        """Binary Classification object from SeriesComparisonRelative object.

        Parameters
        ----------
        comparison : traval.SeriesComparisonRelative
            object comparing two timeseries with base timeseries

        Returns
        -------
        BinaryClassifier
            object for calculating binary classification statistics
        """
        n_true_positives = comparison.idx_r_flagged_in_both.size  # hit
        n_false_positives = comparison.idx_r_flagged_in_s1.size  # false alarm
        n_true_negatives = comparison.idx_r_kept_in_both.size  # correct rejections
        n_false_negatives = comparison.idx_r_flagged_in_s2.size  # miss
        return cls(n_true_positives, n_false_positives,
                   n_true_negatives, n_false_negatives)

    @classmethod
    def from_confusion_matrix(cls, cmat):
        """Create BinaryClassifier from confusion matrix.

        Note
        ----
        Confusion Matrix must be passed as an np.array or pd.DataFrame
        corresponding to: [[TP, FN], [FP, TN]], like the one returned by
        `BinaryClassifier.confusion_matrix`

        Parameters
        ----------
        cmat : np.array or pd.DataFrame
            a 2x2 dataset with structure [[TP, FN],
                                          [FP, TN]]

        Returns
        -------
        BinaryClassifier
            BinaryClassifier object based on values in confusion matrix.

        See also
        --------
        BinaryClassifier.confusion_matrix : for explanation (of abbreviations)
        """
        if isinstance(cmat, pd.DataFrame):
            [tp, fn], [fp, tn] = cmat.values
        elif isinstance(cmat, np.ndarray):
            [tp, fn], [fp, tn] = cmat
        else:
            raise TypeError("Cannot parse confusion matrix of type: "
                            f"{type(cmat)}")
        return cls(tp, fp, tn, fn)

    def __add__(self, other):
        """Add two BinaryClassification objects.

        Parameters
        ----------
        other : traval.BinaryClassifier
            other BinaryClassifier object

        Returns
        -------
        bc : BinaryClassifier
            new BinaryClassifier object containing sum of two objects
        """
        if isinstance(other, self.__class__):
            tp = self.true_positives + other.true_positives
            fp = self.false_positives + other.false_positives
            tn = self.true_negatives + other.true_negatives
            fn = self.false_negatives + other.false_negatives
        else:
            raise TypeError("other must be BinaryClassifier object!")
        return BinaryClassifier(tp, fp, tn, fn)

    def confusion_matrix(self, as_array=False):
        """Calculate confusion matrix.

        Confusion matrix shows the performance of the algorithm given a
        certain truth. An abstract example of the confusion matrix:

                        |     Algorithm     |
                        |-------------------|
                        |  error  | correct |
        ------|---------|---------|---------|
              |  error  |   TP    |   FN    |
        Truth |---------|---------|---------|
              | correct |   FP    |   TN    |
        ------|---------|---------|---------|

        where:
        - TP: True Positives  = errors correctly detected by algorithm
        - TN: True Negatives  = correct values correctly not flagged by algorithm
        - FP: False Positives = correct values marked as errors by algorithm
        - FN: False Negatives = errors not detected by algorithm

        Parameters
        ----------
        as_array : bool, optional
            return data as array instead of DataFrame, by default False

        Returns
        -------
        data : pd.DataFrame or np.array
            confusion matrix
        """

        # create array with data
        data = np.zeros((2, 2), dtype=int)
        # true positives = errors correctly identified
        data[0, 0] = self.true_positives
        # true negatives = correct observations correctly left alone
        data[1, 1] = self.true_negatives
        # false negatives = seen as correct by algorithm but
        # are errors according to 'truth'
        data[0, 1] = self.false_negatives
        # false positives = identified as errors by algorithm but
        # are correct according to 'truth'
        data[1, 0] = self.false_positives

        if as_array:
            return data
        else:
            # create columns and index
            columns = pd.MultiIndex.from_product([["Algorithm"],
                                                  ["error", "correct"]])
            index = pd.MultiIndex.from_product([['"Truth"'],
                                                ["error", "correct"]])
            cmat = pd.DataFrame(
                index=index, columns=columns, data=data, dtype=int)
            return cmat

    @property
    def matthews_correlation_coefficient(self):
        """Matthews correlation coefficient (MCC).

        The MCC is in essence a correlation coefficient between the observed
        and predicted binary classifications; it returns a value between −1
        and +1. A coefficient of +1 represents a perfect prediction, 0 no
        better than random prediction and −1 indicates total disagreement
        between prediction and observation.

        Returns
        -------
        phi : float
            the Matthews correlation coefficient

        See also
        --------
        mcc : convenience method for calculating MCC
        """
        # avoid warning when dividing by 0,
        # returns NaN which is what we want
        with np.errstate(invalid='ignore'):
            phi = ((self.tp * self.tn - self.fp * self.fn) /
                   np.sqrt(float((self.tp + self.fp) * (self.tp + self.fn) *
                                 (self.tn + self.fp) * (self.tn + self.fn))))
        return phi

    @property
    def mcc(self):
        """Convenience method for calculating Matthews correlation coefficient.

        Returns
        -------
        phi : float
            the Matthews correlation coefficient

        See also
        --------
        matthews_correlation_coefficient : more information about the statistic
        """
        return self.matthews_correlation_coefficient

    @property
    def sensitivity(self):
        """Sensitivity or True Positive Rate.

        Statistic describing ratio of true positives identified,
        which also says something about the avoidance of false negatives.

            Sensitivity = TP / (TP + FN)

        where
        - TP : True Positives
        - FN : False Negatives
        """
        tp = self.true_positives
        fn = self.false_negatives
        if tp + fn > 0:
            return tp / (tp + fn)
        else:
            return np.nan

    @property
    def specificity(self):
        """Specificity or True Negative Rate.

        Statistic describing ratio of true negatives identified,
        which also says something about the avoidance of false positives.

            Specificity = TN / (TN + FP)

        where
        - TN : True Negatives
        - FP : False Positives
        """
        tn = self.true_negatives
        fp = self.false_positives
        if tn + fp > 0:
            return tn / (tn + fp)
        else:
            return np.nan

    @property
    def true_positive_rate(self):
        """True Positive Rate. Synonym for sensitivity.

        See sensitiviy for description.
        """
        return self.sensitivity

    @property
    def true_negative_rate(self):
        """True Negative Rate. Synonym for specificity.

        See specificity for description.
        """
        return self.specificity

    @property
    def false_positive_rate(self):
        """False Positive Rate = (1 - specificity).

            FPR = FP / (FP + TN)

        where
        - FP : False Positives
        - TN : True Negatives

        """
        fp = self.false_positives
        tn = self.true_negatives
        if (fp + tn) > 0:
            return fp / (fp + tn)
        else:
            return np.nan

    @property
    def false_negative_rate(self):
        """False Negative Rate = (1 - sensitivity).

            FNR = FN / (FN + TP)

        where
        - FN : False Negatives
        - TP : True Positives

        """
        fn = self.false_negatives
        tp = self.true_positives
        if fn + tp > 0:
            return fn / (fn + tp)
        else:
            return np.nan

    @property
    def informedness(self):
        """Informedness statistic (a.k.a. Youden's J statistic).

        Measure of diagnostic performance, and has a zero value when a
        diagnostic test gives the same proportion of positive results for
        groups with and without a condition, i.e the test is useless.
        A value of 1 indicates that there are no false positives or
        false negatives, i.e. the test is perfect.

        Calculated as:

            informedness = specificity + sensitivity - 1.
        """
        return self.specificity + self.sensitivity - 1.

    @property
    def accuracy(self):
        """Accuracy of binary classification.

            ACC = (TP + TN) / (TP + FP + FN + TN)

        where
        - TP : True Positives
        - TN : True Negatives
        - FP : False Positives
        - FN : False Negatives
        """
        acc = (self.tp + self.tn) / (self.tp + self.fp + self.fn + self.tn)
        return acc

    @property
    def prevalence(self):
        """Prevalance of true errors in total population.

            Prevalence = (TP + FN) / (TP + FP + FN + TN)

        where
        - TP : True Positives
        - TN : True Negatives
        - FP : False Positives
        - FN : False Negatives
        """
        prev = (self.tp + self.fn) / (self.tp + self.fp + self.fn + self.tn)
        return prev

    @property
    def positive_predictive_value(self):
        """Positive predictive value (a.k.a. precision).

            PPV = TP / (TP + FP)

        where
        - TP : True Positives
        - FP : False Positives
        """
        ppv = self.tp / (self.tp + self.fp)
        return ppv

    @property
    def negative_predictive_value(self):
        """Negative predictive value.

            NPV = TN / (TN + FN)

        where
        - TN : True Negatives
        - FN : False Negatives
        """
        if (self.tn + self.fn) > 0:
            npv = self.tn / (self.tn + self.fn)
        else:
            npv = np.nan
        return npv

    @property
    def false_discovery_rate(self):
        """False discovery rate.

            FDR = 1 - PPV = FP / (FP + TP)

        where
        - TP : True Positives
        - FP : False Positives
        """
        fdr = self.fp / (self.fp + self.tp)
        return fdr

    @property
    def false_omission_rate(self):
        """False omission rate.

            FOR = 1 - NPV = FN / (TN + FN)

        where
        - TN : True Negatives
        - FN : False Negatives
        """
        if (self.fn + self.tn) > 0:
            for_ = self.fn / (self.fn + self.tn)
        else:
            for_ = np.nan
        return for_

    def get_all_statistics(self, use_abbreviations=True):
        """Get all statistics in pandas.Series.

        Parameters
        ----------
        use_abbreviations : bool, optional
            whether to use abbreviations or full names for
            index, by default True

        Returns
        -------
        s : pandas.Series
            series containing all statistics
        """

        sdict = {}
        for k, v in self.stats_abbreviations.items():
            if use_abbreviations:
                key = k
            else:
                key = v
            sdict[key] = getattr(self, "_".join(v.split()))

        s = pd.Series(sdict)
        return s
