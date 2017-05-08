class Network( object ):

    def parameters(self): 
        params_w = {'wLyr1': tf.Variable(tf.random_normal([ 3, 3, 1,  self.lyr1FilterNo_                        ])),
                    'wLyr2': tf.Variable(tf.random_normal([ 3, 3,     self.lyr1FilterNo_ , self.lyr2FilterNo_   ])),
                    'wLyr3': tf.Variable(tf.random_normal([ 3, 3,     self.lyr2FilterNo_ , self.lyr3FilterNo_   ])),
                    'wFCh':  tf.Variable(tf.random_normal([ 4* 4*     self.lyr3FilterNo_ , self.fcHidLyrSize_   ])),   
                    'wOut':  tf.Variable(tf.random_normal([           self.fcHidLyrSize_ , self.outLyrSize_     ]))}
                
        params_b = {'bLyr1': tf.Variable(tf.random_normal([           self.lyr1FilterNo_                        ])),
                    'bLyr2': tf.Variable(tf.random_normal([           self.lyr2FilterNo_                        ])),
                    'bLyr3': tf.Variable(tf.random_normal([           self.lyr3FilterNo_                        ])),
                    'bFCh':  tf.Variable(tf.random_normal([           self.fcHidLyrSize_                        ])),
                    'bOut':  tf.Variable(tf.random_normal([           self.outLyrSize_                          ]))}
        return params_w,params_b

    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def score(self):

        def conv2d(x, W, b, strides=1):
            x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
            x = tf.nn.bias_add(x, b)
            return x

        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------  

        def maxpool2d(x, k=2):
            return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')
        
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        
        self.x = tf.reshape(x, shape = [-1,32,32,1])
        
        # 1)  
        convLyr_1_conv = conv2d (self.x, self.params_w_['wLyr1'], self.params_b_['bLyr1'])
        convLyr_1_relu = tf.nn.relu(convLyr_1_conv) 
        convLyr_1_pool = maxpool2d(convLyr_1_relu, k=2)
        
        # 2)
        convLyr_2_conv = conv2d(convLyr_1_pool, self.params_w_['wLyr2'], self.params_b_['bLyr2'])
        convLyr_2_relu = tf.nn.relu(convLyr_2_conv)
        convLyr_2_pool = maxpool2d(convLyr_2_relu, k=2)

        # 3)
        convLyr_3_conv = conv2d(convLyr_2_pool, self.params_w_['wLyr3'], self.params_b_['bLyr3'])
        convLyr_3_relu = tf.nn.relu(convLyr_3_conv)
        convLyr_3_pool = maxpool2d(convLyr_3_relu, k=2)
        
        # 4) Fully Connected
        fcLyr_1 = tf.reshape(convLyr_3_pool, [-1,self.params_w_['wFCh'].get_shape().as_list()[0]])
        fcLyr_1 = tf.add(tf.matmul(fcLyr_1, self.params_w_['wFCh']), self.params_b_['bFCh'])
        fcLyr_1 = tf.nn.relu(fcLyr_1)
        fcLyr_1 = tf.nn.dropout(fcLyr_1, self.keepProb)
        
        netOut = tf.add(tf.matmul(fcLyr_1, self.params_w_['wOut']), self.params_b_['bOut'])
        
        return netOut  
        
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ 
    
    def largeCost(self): 
        maxScore    = tf.reduce_max( self.score ,[1] ) 
        temp        = self.score 
        temp       -= tf.transpose(maxScore)
        total_loss  = tf.reduce_mean(-tf.reduce_sum ( self.y * temp , [1] ) )   
        return total_loss 
    
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    def smallCost(self):   
        maxScore   = tf.reduce_max( self.score ,[1] ) 
        temp       = self.score 
        temp      -= tf.transpose(maxScore) 
        softed     = tf.nn.softmax(temp) 
        total_loss = tf.reduce_mean( -tf.reduce_sum ( self.y * tf.log( softed ), [1] ) )   
        return total_loss 
    
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    def cost(self):  
        return tf.cond( tf.greater( tf.reduce_max(self.score) , self.threshold ), self.largeCost , self.smallCost )  
        
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------   

    def optimizer(self):
        return tf.train.AdamOptimizer(learning_rate = self.lr).minimize(self.cost)

    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def accuracy(self):
        correct_pred = tf.equal(tf.argmax(self.score,1), tf.argmax(y,1))
        return(tf.reduce_mean(tf.cast(correct_pred, tf.float32)))
        
   #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------    

    def __init__( self, x, y, lr, lyr1FilterNo, lyr2FilterNo, lyr3FilterNo, fcHidLyrSize, inLyrSize, outLyrSize, keepProb, threshold ):
        
        self.x             = x
        self.y             = y
        self.lr            = lr
        self.inLyrSize     = inLyrSize
        self.outLyrSize_   = outLyrSize
        self.lyr1FilterNo_ = lyr1FilterNo
        self.lyr2FilterNo_ = lyr2FilterNo
        self.lyr3FilterNo_ = lyr3FilterNo
        self.fcHidLyrSize_ = fcHidLyrSize
        self.keepProb      = keepProb
        self.threshold     = threshold

        [self.params_w_, self.params_b_] = Network.parameters (self) # initialization and packing the parameters 
        self.score                       = Network.score      (self) # Computing the score function
        self.cost                        = Network.cost       (self) # Computing the cost function
        self.optimizer                   = Network.optimizer  (self) # Computing the update function
        self.accuracy                    = Network.accuracy   (self) # performance
