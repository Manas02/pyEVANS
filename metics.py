import numpy as np
from scipy import linalg, special, stats


def check_size(y_pred, y_true):
    if len(y_true) != len(y_pred):
        raise Exception(f'The sizes of arrays are not the same size \n \
                y_pred size:{len(y_pred)}\ny_true size:{len(y_true)}')


def mean_absolute_error(y_pred, y_true):
    check_size(y_pred, y_true)
    return np.average(np.abs(y_pred - y_true))


def mean_squared_error(y_pred, y_true):
    check_size(y_pred, y_true)
    return np.sum(np.square(y_pred - y_true)) / len(y_pred)


def root_mean_squared_error(y_pred, y_true):
    check_size(y_pred, y_true)
    return np.sqrt(np.sum(np.square(y_pred - y_true)) / len(y_pred))


def accuracy_score(y_true, y_pred, normalize=True):
    """Accuracy classification score.
    In multilabel classification, this function computes subset accuracy:
    the set of labels predicted for a sample must *exactly* match the
    corresponding set of labels in y_true.
    Read more in the :ref:`User Guide <accuracy_score>`.
    
    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) labels.
    y_pred : 1d array-like, or label indicator array / sparse matrix
        Predicted labels, as returned by a classifier.
    normalize : bool, default=True
        If ``False``, return the number of correctly classified samples.
        Otherwise, return the fraction of correctly classified samples.
    
    Returns
    -------
    score : float
        If ``normalize == True``, return the fraction of correctly
        classifid samples (float), else returns the number of correctly
        classified samples (int).
        The best performance is 1 with ``normalize == True`` and the number
        of samples with ``normalize == False``.
    """
    check_size(y_pred, y_true)
    if normalize:
        return sum((y_pred - y_true) == 0) / len(y_true)
    else:
        return sum((y_pred - y_true) == 0)


def pearson_correlation(x, y):
    """
    Pearson correlation coefficient and p-value for testing non-correlation.

    The Pearson correlation coefficient [1]_ measures the linear relationship
    between two datasets.  The calculation of the p-value relies on the
    assumption that each dataset is normally distributed.  (See Kowalski [3]_
    for a discussion of the effects of non-normality of the input on the
    distribution of the correlation coefficient.)  Like other correlation
    coefficients, this one varies between -1 and +1 with 0 implying no
    correlation. Correlations of -1 or +1 imply an exact linear relationship.
    Positive correlations imply that as x increases, so does y. Negative
    correlations imply that as x increases, y decreases.

    The p-value roughly indicates the probability of an uncorrelated system
    producing datasets that have a Pearson correlation at least as extreme
    as the one computed from these datasets.

    Parameters
    ----------
    x : (N,) array_like
        Input array.
    y : (N,) array_like
        Input array.

    Returns
    -------
    r : float
        Pearson's correlation coefficient.
    p-value : float
        Two-tailed p-value.

    .. math::

        r = \frac{\sum (x - m_x) (y - m_y)}
                 {\sqrt{\sum (x - m_x)^2 \sum (y - m_y)^2}}

    where :math:`m_x` is the mean of the vector :math:`x` and :math:`m_y` is
    the mean of the vector :math:`y`.

    Under the assumption that :math:`x` and :math:`m_y` are drawn from
    independent normal distributions (so the population correlation coefficient
    is 0), the probability density function of the sample correlation
    coefficient :math:`r` is ([1]_, [2]_):

    .. math::

        f(r) = \frac{{(1-r^2)}^{n/2-2}}{\mathrm{B}(\frac{1}{2},\frac{n}{2}-1)}

    where n is the number of samples, and B is the beta function.  This
    is sometimes referred to as the exact distribution of r.  This is
    the distribution that is used in `pearsonr` to compute the p-value.
    The distribution is a beta distribution on the interval [-1, 1],
    with equal shape parameters a = b = n/2 - 1.  In terms of SciPy's
    implementation of the beta distribution, the distribution of r is::

        dist = scipy.stats.beta(n/2 - 1, n/2 - 1, loc=-1, scale=2)

    The p-value returned by `pearsonr` is a two-sided p-value.  For a
    given sample with correlation coefficient r, the p-value is
    the probability that abs(r') of a random sample x' and y' drawn from
    the population with zero correlation would be greater than or equal
    to abs(r).  In terms of the object ``dist`` shown above, the p-value
    for a given r and length n can be computed as::

        p = 2*dist.cdf(-abs(r))

    When n is 2, the above continuous distribution is not well-defined.
    One can interpret the limit of the beta distribution as the shape
    parameters a and b approach a = b = 0 as a discrete distribution with
    equal probability masses at r = 1 and r = -1.  More directly, one
    can observe that, given the data x = [x1, x2] and y = [y1, y2], and
    assuming x1 != x2 and y1 != y2, the only possible values for r are 1
    and -1.  Because abs(r') for any sample x' and y' with length 2 will
    be 1, the two-sided p-value for a sample of length 2 is always 1.

    References
    ----------
    .. [1] "Pearson correlation coefficient", Wikipedia,
           https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
    .. [2] Student, "Probable error of a correlation coefficient",
           Biometrika, Volume 6, Issue 2-3, 1 September 1908, pp. 302-310.
    .. [3] C. J. Kowalski, "On the Effects of Non-Normality on the Distribution
           of the Sample Product-Moment Correlation Coefficient"
           Journal of the Royal Statistical Society. Series C (Applied
           Statistics), Vol. 21, No. 1 (1972), pp. 1-12.
    """
    n = len(x)
    check_size(x, y)

    if n < 2:
        raise ValueError('x and y must have length at least 2.')


    x = np.asarray(x)
    y = np.asarray(y)


    # If an input is constant, the correlation coefficient is not defined.
    if (x == x[0]).all() or (y == y[0]).all():
        print("input is constant, the correlation coefficient is not defined.")
        return np.nan, np.nan


    # dtype is the data type for the calculations.  This expression ensures
    # that the data type is at least 64 bit floating point.  It might have
    # more precision if the input is, for example, np.longdouble.
    dtype = type(1.0 + x[0] + y[0])


    if n == 2:
        return dtype(np.sign(x[1] - x[0]) * np.sign(y[1] - y[0])), 1.0


    xmean = x.mean(dtype=dtype)
    ymean = y.mean(dtype=dtype)


    # By using `astype(dtype)`, we ensure that the intermediate calculations
    # use at least 64 bit floating point.
    xm = x.astype(dtype) - xmean
    ym = y.astype(dtype) - ymean


    # Unlike np.linalg.norm or the expression sqrt((xm*xm).sum()),
    # scipy.linalg.norm(xm) does not overflow if xm is, for example,
    # [-5e210, 5e210, 3e200, -3e200]
    normxm = linalg.norm(xm)
    normym = linalg.norm(ym)


    threshold = 1e-13
    if normxm < threshold * abs(xmean) or normym < threshold * abs(ymean):
        # If all the values in x (likewise y) are very close to the mean,
        # the loss of precision that occurs in the subtraction xm = x - xmean
        # might result in large errors in r.
        raise Exception("input is constant, the correlation coefficient is not defined.")


    r = np.dot(xm/normxm, ym/normym)


    # Presumably, if abs(r) > 1, then it is only some small artifact of
    # floating point arithmetic.
    r = max(min(r, 1.0), -1.0)


    # As explained in the docstring, the p-value can be computed as
    #     p = 2*dist.cdf(-abs(r))
    # where dist is the beta distribution on [-1, 1] with shape parameters
    # a = b = n/2 - 1.  `special.btdtr` is the CDF for the beta distribution
    # on [0, 1].  To use it, we make the transformation  x = (r + 1)/2; the
    # shape parameters do not change.  Then -abs(r) used in `cdf(-abs(r))`
    # becomes x = (-abs(r) + 1)/2 = 0.5*(1 - abs(r)).  (r is cast to float64
    # to avoid a TypeError raised by btdtr when r is higher precision.)
    ab = n / 2 - 1
    prob = 2 * special.btdtr(ab, ab, 0.5 * (1 - abs(np.float64(r))))

    return r, prob


def bootstrap(x, alpha=0.05):
    """
    References
    ----------
    .. [1] "Bootstrap Confidence Intervals",
           http://www2.stat.duke.edu/~ar182/rr/examples-gallery/BootstrapConfidenceIntervals.html
    """
    sd = np.sqrt(np.sum(np.power(x - x.mean(),2)) / (x.size-1))
    interval = stats.t.ppf(1.0 - (alpha / 2.0),x.size-1) * (sd / np.sqrt(x.size))
    confidence_interval = (x.mean() - interval, x.mean() + interval)
    return confidence_interval


def f_test(x, y, alpha=0.05):
    f = np.var(x, ddof = 1) / np.var(y, ddof = 1)
    dfn = x.size - 1 
    dfd = y.size - 1 
    p = 1 - stats.f.cdf(f, dfn, dfd) 
    print(f'p-value ({p}) is less than {alpha}, null hypothesis is rejected') if p < alpha else None
    return f, p

