This is only show that how the 'softmax_cross_entropy_with_logits()' function in Tensorflow internally works without numerically unstable problem and estimate the exp() of large numbers.
 
The "Numerically-Stable-Cross-Entropy-SingleLabel.py" file represent the cost function for single label problems and "Numerically-Stable-Cross-Entropy-MultiLabel.py" represents the cost function for multi-label (specificly two label) problems.

### Numerically Stable Cross Entropy Loss Function:
Since the large numbers in `exp()` functoin of python returns 'inf' (more than 709 in python 2.7.11), so in these version of cross entropy loss without 'softmax_cross_entropy_with_logits()' function, I used a condition of checking the highest value in logits, which is determined by `threshold` variable in code.

### Accuracy Function in Multi-label Task:

Note that in the multi-label problems since the calculating accuracy is a littile bit different that ordinary way, in the "Numerically-Stable-Cross-Entropy-MultiLabel.py", 'perfFun()' function returns two boolean tensor each represents the accuracy in one dimension of multi-label task, you should merge these two boolean tensor inside the session to calculate the final accuracy.


### Notes on 'softmax_cross_entropy_with_logits()' Function in Tensorflow:

Actually this function doesn't calculate the exact loss (in large numbers), and only approximate it. Let's try this simple code in python and check the results:

```python
import numpy      as np
import tensorflow as tf

if __name__ == '__main__':

    logits = np.array([1000, 2000, 2500], dtype=np.float32)
    labels = np.array([0   , 1   , 0   ], dtype=np.float32)
    
    with tf.Session() as sess:
        print (tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits( logits, labels ))).eval()
        
        softed = tf.nn.softmax(logits)
        print (tf.reduce_mean(-tf.reduce_sum(labels * tf.log(softed), reduction_indices=[0]))).eval()
```

It prints `500.0` for first one and `nan` for second one,

### Problem of Numerically Unstable:

Based on Tensorflow document in [here](https://www.tensorflow.org/get_started/mnist/beginners#training) without using the 'softmax_cross_entropy_with_logits()' function for calculating loss in Tensorflow, we face the problem of numerically unstable results,
actually happen in large numbers, this problem arises when the logits from the network output are large numbers, so python returns 'inf' in result, consider our network has 3 output, and they are large numbers such: [1000, 2000, 2500], now we should sqush this logits with Softmax function to have probabilities:

- p1 = exp(1000) / exp(1000) + exp(2000) + exp(2500)
- p2 = exp(2000) / exp(1000) + exp(2000) + exp(2500)
- p2 = exp(2500) / exp(1000) + exp(2000) + exp(2500)

Since python (more specificly python version 2.7.11) returns 'inf' (infinity) for values more than 709 in exp() function. So for example in case p2 we have:

- divide numerator and denominator by exp value of numerator,
- then we have 1 / exp(-1000) + 1 + exp(1500)
- in this case the numerator is 1 and denominator is very large number due exp(1500)
- the result will be very very small number like 1.E-50 (aproximatly), which can't display with any variable in python, so it will be saved as zero.
- in case p2 if we divide numerator and denominator by 2500 so we will have: 1 / 1 + a + b, a and b are very small numbers, so the result will be aproximatly 0.99999...
- the result after softmax normalizing will be: predictions= [0, 0, 1]
- consider the middle one as 1 in ground truth: labels= [0, 1, 0]
- now we should calculate the cross-entropy loss
- cross-entropy loss is equal:  - sum( labels * tf.log(predictions) ) / n
- so we have: 0.log(0) + 1.log(0) + 0.log(1) = 0 + NaN + 0 = NaN (Not a Number)
- this NaN value as cost cause network learn nothing


