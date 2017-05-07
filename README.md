This is only show that how the 'softmax_cross_entropy_with_logits()' function in Tensorflow internally works without numerically unstable problem and estimate the exp() of large numbers.


Based on Tensorflow document (link to mnist example) without using the 'softmax_cross_entropy_with_logits()' function for calculating loss in Tensorflow, we face the problem of numerically unstable results, actually happen in large numbers, this problem arises when the logits from the network output are large numbers, so python returns 'inf' in result, consider our network has 3 output, and they are large numbers such: [1000, 2000, 2500], now we should sqush this logits with Softmax function to have probabilities:

- p1 = exp(1000) / exp(1000) + exp(2000) + exp(2500)
- p2 = exp(2000) / exp(1000) + exp(2000) + exp(2500)
- p2 = exp(2500) / exp(1000) + exp(2000) + exp(2500)

Since python (more specificly python version 2.7.11) returns 'inf' (infinity) for values more than 709 in exp() function. So for example in case p2 we have:

- divide numerator and denominator by exp value of numerator,
- then we have 1 / exp(-1000) + 1 + exp(1500)
- in this case the numerator is 1 and denominator is very large number due exp(1500)
- the result will be very very small number like 1.E-50 (aproximatly), which can't display with any variable in python, so it will be saved as zero.

 
The "Numerically-Stable-Cross-Entropy-SingleLabel.py" file  

