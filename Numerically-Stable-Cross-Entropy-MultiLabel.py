class ConvNet( object ):

    def paramsFun(self): 
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

    def scoreFun(self):

        def conv2d(x, w, b, strides=1): 
            x = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding='SAME')
            x = tf.nn.bias_add(x, b) 
            return x

        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------  

        def maxpool2d(x, k=2):
            return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')
        
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        
        self.x_ = tf.reshape(x, shape = [-1,32,32,1])
        
        # 1)   
        convLyr_1_conv = conv2d (self.x_, self.params_w_['wLyr1'], self.params_b_['bLyr1'])
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
        fcLyr_1 = tf.nn.dropout(fcLyr_1, self.keepProb_)  
        
        netOut = tf.add(tf.matmul(fcLyr_1, self.params_w_['wOut']), self.params_b_['bOut']) 
        
        return netOut 
        
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ 
    
    def largeCost(self):
        score_split = tf.split(1, 2, self.score_ )  
        for i in range(len(score_split)):
            maxScore = tf.reduce_max( score_split[i] ,[1] )
            maxScore = tf.reshape(maxScore, shape = (300, 1)) # 300 is batch_size
            score_split[i] -= maxScore 
        all_in_one  = tf.concat(1, score_split) 
        total_loss  = tf.reduce_mean(-tf.reduce_sum ( self.y_  * all_in_one , [1] ) )    
        return  total_loss 
    
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    def smallCost(self):  
        score_split           = tf.split(1, 2, self.score_)  
        for i in range(len(score_split)):
            score_split[i]   -= tf.reduce_max(score_split[i])  
        score_split_softmaxed = [tf.nn.softmax(c) for c in score_split]  
        all_in_one_softmaxed  = tf.concat(1, score_split_softmaxed) 
        total_loss            = tf.reduce_mean(-tf.reduce_sum (self.y_  * tf.log( all_in_one_softmaxed ), [1] ) )   
        return total_loss 
    
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    def costFun(self):  
        total_loss = tf.cond( tf.greater( tf.reduce_max(self.score_) , self.threshold_ ), self.largeCost , self.smallCost )  
        tf.summary.scalar("cost", total_loss)
        return total_loss
        
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def costFun2(self):
        score_split = tf.split( 1, 2, self.score_ )
        label_split = tf.split( 1, 2, self.y_     ) 
        total = 0.0
        for i in range ( len(score_split) ): 
            total += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits( score_split[i] , label_split[i] ))    
        return total
        
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------   

    def updateFun(self):
        return tf.train.AdamOptimizer(learning_rate = self.lr_).minimize(self.cost_)

    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def perfFun(self): 
    
        score_split = tf.split( 1, 2, self.score_ )
        label_split = tf.split( 1, 2, self.y_     ) 
        
        correct_pred1  = tf.equal(tf.argmax(score_split[0],1), tf.argmax(label_split[0],1))  
        correct_pred2  = tf.equal(tf.argmax(score_split[1],1), tf.argmax(label_split[1],1))  
        
        return correct_pred1 , correct_pred2
    
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    def sumFun(self):
        return tf.summary.merge_all()    
        
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------    

    def __init__(self,x,y,lr,lyr1FilterNo,lyr2FilterNo,lyr3FilterNo,fcHidLyrSize,inLyrSize,outLyrSize, keepProb, threshold):
        
        self.x_            = x
        self.y_            = y
        self.lr_           = lr
        self.inLyrSize     = inLyrSize
        self.outLyrSize_   = outLyrSize
        self.lyr1FilterNo_ = lyr1FilterNo
        self.lyr2FilterNo_ = lyr2FilterNo
        self.lyr3FilterNo_ = lyr3FilterNo
        self.fcHidLyrSize_ = fcHidLyrSize
        self.keepProb_     = keepProb
        self.threshold_    = threshold

        [self.params_w_, self.params_b_] = ConvNet.paramsFun(self) # initialization and packing the parameters
        self.score_                      = ConvNet.scoreFun (self) # Computing the score function
        self.largeCost_                  = ConvNet.largeCost(self)
        self.smallCost_                  = ConvNet.smallCost(self)
        self.cost_                       = ConvNet.costFun  (self) # Computing the cost function
        self.cost_2                      = ConvNet.costFun2 (self) # Computing the cost function
        self.update_                     = ConvNet.updateFun(self) # Computing the update function
        self.perf_1, self.perf_2         = ConvNet.perfFun  (self) # performance 
