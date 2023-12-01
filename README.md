This is only show that how the 'softmax_cross_entropy_with_logits()' function in Tensorflow internally works without numerically unstable problem and estimate the exp() of large numbers.
 
The "Numerically-Stable-Cross-Entropy-SingleLabel.py" file represent the cost function for single label problems and "Numerically-Stable-Cross-Entropy-MultiLabel.py" represents the cost function for multi-label (specificly two label) problems.

### Numerically Stable Cross Entropy Loss Function:
Since the large numbers in `exp()` function of python returns 'inf' (more than 709 in python 2.7.11), so in these version of cross entropy loss without 'softmax_cross_entropy_with_logits()' function, I used a condition of checking the highest value in logits, which is determined by `threshold` variable in code. For larger scores in logit it use to approximate the loss, but for smaller scores it use ordinary way to calculate loss.

### Notes on 'softmax_cross_entropy_with_logits()' Function in Tensorflow:

Actually this function doesn't calculate the exact loss (in large numbers), and only approximate it. Let's try this simple code in python and check the results:

```python
import numpy      as np
import tensorflow as tf

if __name__ == '__main__':

    logits = np.array([1000, 2000, 2500], dtype=np.float32)
    labels = np.array([0   , 1   , 0   ], dtype=np.float32)
    
    with tf.Session() as sess:
        print (tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits( logits=logits, labels=labels )).eval())

        softed = tf.nn.softmax(logits)
        print (tf.reduce_mean(-tf.reduce_sum(labels * tf.log(softed), reduction_indices=[0])).eval())
```

It prints `500.0` for the first one and `nan` for the second one, as you can see it doesn't calculate the exact loss value, only approximately return it. The approach is very simple, actually is reduce every score from the max score, so in this case [1000, 2000, 2500], after reducing 2500 we have [-1500, -500, 0], then it uses this values without squashing them with Softmax, note that the negative values will be removed with negative sign in the formula: `tf.reduce_mean(-tf.reduce_sum(labels * tf.log(softed), reduction_indices=[0]))`, and this is the method I use in my code.

### Problem of Numerically Unstable:

Based on Tensorflow document in [here](https://www.tensorflow.org/get_started/mnist/beginners#training) without using the 'softmax_cross_entropy_with_logits()' function for calculating loss in Tensorflow, we face the problem of numerically unstable results,
actually happen in large numbers, this problem arises when the logits from the network output are large numbers, so python returns 'inf' in result, consider our network has 3 output, and they are large numbers such: [1000, 2000, 2500], now we should sqush this logits with Softmax function to have probabilities:

- p1 = exp(1000) / exp(1000) + exp(2000) + exp(2500)
- p2 = exp(2000) / exp(1000) + exp(2000) + exp(2500)
- p3 = exp(2500) / exp(1000) + exp(2000) + exp(2500)

Since python (specifically python version 2.7.11) returns 'inf' (infinity) for values more than 709 in exp() function, i.e, in case p2 we have:

- Divide numerator and denominator by exp value of numerator,
- Then we have 1 / exp(-1000) + 1 + exp(1500)
- In this case the numerator is 1 and denominator is very large number due exp(1500)
- The result will be very very small number number like 1.E-50 (approximately), which will be saved as zero.
- In case p3 if we divide numerator and denominator by 2500 so we will have: 1 / 1 + a + b, a and b are very small numbers, so the result will be approximately 0.99999...
- The result after softmax normalizing will be: predictions= [0, 0, 1]
- Consider the middle one as 1 in ground truth: labels= [0, 1, 0]
- Now we should calculate the cross-entropy loss
- Cross-entropy loss is equal:  -sum( labels * tf.log(predictions) ) / n
- So we have: 0.log(0) + 1.log(0) + 0.log(1) = 0 + NaN + 0 = NaN (Not a Number)
- This NaN value as cost cause network learn nothing

### Accuracy Function in Multi-label Task:

Note that in the multi-label problems since the calculating accuracy is a little bit different that ordinary way, in the "Numerically-Stable-Cross-Entropy-MultiLabel.py", 'perfFun()' function returns two boolean tensor each represents the accuracy in one dimension of multi-label task, you should merge these two boolean tensor inside the session to calculate the final accuracy.

### 'costFun2' inside "Numerically-Stable-Cross-Entropy-MultiLabel.py":

There is two cost function for multi-label classification task, the second one use 'softmax_cross_entropy_with_logits()' function, and you can see both of functions compute the same cost value as result.
