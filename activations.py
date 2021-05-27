import numpy as np
from math import erf


def softmax(x, axis=None):
    """Softmax converts a tensor of values to a probability distribution.

    The elements of the output tensor are in range (0, 1) and sum to 1.

    Each tensor is handled independently. The `axis` argument sets which axis
    of the input the function is applied along.
    axis 1 represents column and axis 0 represents row
    Softmax is often used as the activation for the last
    layer of a classification network because the result could be interpreted as
    a probability distribution.

    The softmax of each tensorx is computed as
    `exp(x) / sum(exp(x))`

    Args:
        x    : Input tensor.
        axis : Integer, axis along which the softmax normalization is applied.

    Returns:
        vector, output of softmax transformation (all values are non-negative
        and sum to 1).
    """
    return np.exp(x) / np.sum(np.exp(x), axis=axis)


def elu(x, alpha=1.0):
    """Exponential Linear Unit.
    The exponential linear unit (ELU) with `alpha > 0` is:
    `x` if `x > 0` and
    `alpha * (exp(x) - 1)` if `x < 0`
    The ELU hyperparameter `alpha` controls the value to which an
    ELU saturates for negative net inputs. ELUs diminish the
    vanishing gradient effect.
    ELUs have negative values which pushes the mean of the activations
    closer to zero.
    Mean activations that are closer to zero enable faster learning as they
    bring the gradient closer to the natural gradient.
    ELUs saturate to a negative value when the argument gets smaller.
    Saturation means a small derivative which decreases the variation
    and the information that is propagated to the next layer.
  
    Args:
        x     : Input tensor.
        alpha : A scalar, slope of negative section. `alpha` controls the value to
        which an ELU saturates for negative net inputs.

    Returns:
        The exponential linear unit (ELU) activation function: `x` if `x > 0` and
        `alpha * (exp(x) - 1)` if `x < 0`.
 
    Reference:
        [Fast and Accurate Deep Network Learning by Exponential Linear Units
        (ELUs) (Clevert et al, 2016)](https://arxiv.org/abs/1511.07289)
    """
    return x if x >= 0 else alpha * (np.exp(x) - 1)


def selu(x, alpha=1.67326324 ,scale=1.05070098):
    """Scaled Exponential Linear Unit (SELU).
    The Scaled Exponential Linear Unit (SELU) activation function is defined as:
    - `if x > 0: return scale * x`
    - `if x < 0: return scale * alpha * (exp(x) - 1)`
    where `alpha` and `scale` are pre-defined constants
    (`alpha=1.67326324` and `scale=1.05070098`).
    Basically, the SELU activation function multiplies `scale` (> 1) with the
    output of the `elu` function to ensure a slope larger than 
    one for positive inputs.
    The values of `alpha` and `scale` are
    chosen so that the mean and variance of the inputs are preserved
    between two consecutive layers as long as the weights are initialized
    correctly and the number of input units is "large enough"
    (see reference paper for more information).

    Args:
        x: A tensor variable to compute the activation function for.
  
    Returns:
        The scaled exponential unit activation: `scale * elu(x, alpha)`.
  
    References:
        - [Klambauer et al., 2017](https://arxiv.org/abs/1706.02515)
    """
    return scale * x if x >= 0 else scale * alpha * (np.exp(x) - 1)


def softplus(x):
    """Softplus activation function, 
    `softplus(x) = log(exp(x) + 1)`.
  
    Args:
        x: Input tensor.
  
    Returns:
        The softplus activation: `log(exp(x) + 1)`.
    """
    return np.log(np.exp(x) + 1)


def softsign(x):
    """Softsign activation function, 
    `softsign(x) = x / (abs(x) + 1)`.
  
    Args:
        x: Input tensor.
  
    Returns:
        The softsign activation: `x / (abs(x) + 1)`.
    """
    return x / (np.absolute(x) + 1)


def swish(x):
    """Swish activation function, 
    `swish(x) = x * sigmoid(x)`.
    Swish activation function which returns `x * sigmoid(x)`.
    It is a smooth, non-monotonic function that consistently matches
    or outperforms ReLU on deep networks, it is unbounded above and
    bounded below.
  
    Args:
        x: Input tensor.
  
    Returns:
        The swish activation applied to `x` (see reference paper for details).
  
    Reference:
        - [Ramachandran et al., 2017](https://arxiv.org/abs/1710.05941)
    """
    return x * sigmoid(x)


def relu(x, alpha=0., max_value=None, threshold=0):
    """Applies the rectified linear unit activation function.
    With default values, this returns the standard ReLU activation:
    `max(x, 0)`, the element-wise maximum of 0 and the input tensor.
    Modifying default parameters allows you to use non-zero thresholds,
    change the max value of the activation,
    and to use a non-zero multiple of the input for values below the threshold.
  
    Args:
        x         : Input `tensor` or `variable`.
        alpha     : A `float` that govern the slope for values lower than the
                    threshold.
        max_value : A `float` that sets the saturation threshold (the largest value
                    the function will return).
        threshold : A `float` giving the threshold value of the activation function
                    below which values will be damped or set to zero.
  
    Returns:
        A `Tensor` representing the input tensor,
        transformed by the relu activation function.
        Tensor will be of the same shape and dtype of input `x`.
    """
    return np.maximum(x, 0 if max_value == None else max_value)


def gelu(x, approximate=False):
    """Applies the Gaussian error linear unit (GELU) activation function.
    Gaussian error linear unit (GELU) computes
    `x * P(X <= x)`, where `P(X) ~ N(0, 1)`.
    The (GELU) nonlinearity weights inputs by their value, rather than gates
    inputs by their sign as in ReLU.
  
    Args:
        x           : Input tensor.
        approximate : A `bool`, whether to enable approximation.

    Returns:
        The gaussian error linear activation:
        `0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))`
        if `approximate` is `True` or
        `x * P(X <= x) = 0.5 * x * (1 + erf(x / sqrt(2)))`,
        where `P(X) ~ N(0, 1)`,
        if `approximate` is `False`.
  
    Reference:
        - [Gaussian Error Linear Units (GELUs)](https://arxiv.org/abs/1606.08415)
    """
    if approximate == True:
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * (x ** 3)))) 
    elif approximate == False:
        return 0.5 * x * (1 + erf(x / np.sqrt(2)))


def tanh(x):
    """Hyperbolic tangent activation function.
  
    Args:
        x: Input tensor.
  
    Returns:
      Tensor of same shape and dtype of input `x`, with tanh activation:
      `tanh(x) = sinh(x)/cosh(x) = ((exp(x) - exp(-x))/(exp(x) + exp(-x)))`.
    """
    return np.tanh(x)


def sigmoid(x):
    """Sigmoid activation function, `sigmoid(x) = 1 / (1 + exp(-x))`.
    Applies the sigmoid activation function. For small values (<-5),
    `sigmoid` returns a value close to zero, and for large values (>5)
    the result of the function gets close to 1.
    Sigmoid is equivalent to a 2-element Softmax, where the second element is
    assumed to be zero. The sigmoid function always returns a value between
    0 and 1.
    
    Args:
        x: Input tensor.
    
    Returns:
        Tensor with the sigmoid activation: `1 / (1 + exp(-x))`.
    """
    return 1 / (1 + np.exp(-x))


def exponential(x):
    """Exponential activation function.
  
    Args:
        x: Input tensor.
  
    Returns:
        Tensor with exponential activation: `exp(x)`.
    """
    return np.exp(x)


def hard_sigmoid(x):
    """Hard sigmoid activation function.
    A faster approximation of the sigmoid activation.
    Piecewise linear approximation of the sigmoid function.
    Ref: 'https://en.wikipedia.org/wiki/Hard_sigmoid'
  
    Args:
        x: Input tensor.
  
    Returns:
        The hard sigmoid activation, defined as:
        - `if x < -2.5: return 0`
        - `if x > 2.5: return 1`
        - `if -2.5 <= x <= 2.5: return 0.2 * x + 0.5`
    """
    if x < -2.5 :
        return 0
    elif x > 2.5 :
        return 1
    else:
        return 0.2 * x + 0.5


def linear(x):
    """Linear activation function (pass-through).
  
    Args:
        x: Input tensor.
    
    Returns:
        The input, unmodified.
    """
    return x


def leaky_relu(x, alpha=0.2):
    """Compute the Leaky ReLU activation function.
    Source: [Rectifier Nonlinearities Improve Neural Network Acoustic Models.
    AL Maas, AY Hannun, AY Ng - Proc. ICML, 2013]
    (https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf).
  
    Args:
        x        : A `Tensor` representing preactivation values. Must be one of
                  the following types: `float16`, `float32`, 
                  `float64`, `int32`, `int64`.
        alpha    : Slope of the activation function at x < 0.
  
    Returns:
        The activation value.
    
    References:
        Rectifier Nonlinearities Improve Neural Network Acoustic Models:
            [Maas et al., 2013]
        (http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.693.1422)
        ([pdf]
        (http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.693.1422&rep=rep1&type=pdf))
    """
    return np.maximum(0, x) + alpha * np.minimum(0, x)


def log_softmax(x):
    # FIXME: Seems to produce incorrect result.
    return np.log(softmax(x))
    # !!!!!!!!!!!!DOES NOT GIVE CORRECT RETURN!!!!!!!!
