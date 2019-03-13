
def initialise_parameter(n_x,n_y):
    
    # defining parameters
    input_feat = n_x
    hidden1_nodes = 50
    hidden2_nodes = 100
    hidden3_nodes = 50
    output_feat = n_y
    
    # defining variable initializer
    var_init = tf.variance_scaling_initializer()
    
    with tf.variable_scope('weights'):
        # defining weights and bias
        W1 = tf.Variable(var_init([input_feat,hidden1_nodes]),dtype=tf.float32,name='w1')
        W2 = tf.Variable(var_init([hidden1_nodes,hidden2_nodes]),dtype=tf.float32,name='w2')
        W3 = tf.Variable(var_init([hidden2_nodes,hidden3_nodes]),dtype=tf.float32,name='w3')
        W4 = tf.Variable(var_init([hidden3_nodes,output_feat]),dtype=tf.float32,name='w4')
        
        W_AE = tf.Variable(var_init([hidden1_nodes,input_feat]),dtype=tf.float32,name='w_ae')
    
    with tf.variable_scope('bias'):
        b1 = tf.Variable(tf.zeros(hidden1_nodes),name='b1')
        b2 = tf.Variable(tf.zeros(hidden2_nodes),name='b2')
        b3 = tf.Variable(tf.zeros(hidden3_nodes),name='b3')
        b4 = tf.Variable(tf.zeros(output_feat),name='b4')
        
        B_AE = tf.Variable(tf.zeros(input_feat),name='b_ae')

    parameters = {            
            'W1':W1,
            'W2':W2,
            'W3':W3,
            'W4':W4,
            'W_AE':W_AE,
            'b1':b1,
            'b2':b2,
            'b3':b3,
            'b4':b4,
            'B_AE':B_AE}
    return parameters


def initialise_parameter_ae_all(n_x,n_y):
    ''' all layers of the FC is trained by ae '''
    # defining parameters
    input_feat = n_x
    hidden1_nodes = 50
    hidden2_nodes = 100
    hidden3_nodes = 50
    output_feat = n_y
    
    # defining variable initializer
    var_init = tf.variance_scaling_initializer()
    
    with tf.variable_scope('weights'):
        # defining weights and bias
        W1 = tf.Variable(var_init([input_feat,hidden1_nodes]),dtype=tf.float32,name='w1')
        W2 = tf.Variable(var_init([hidden1_nodes,hidden2_nodes]),dtype=tf.float32,name='w2')
        W3 = tf.Variable(var_init([hidden2_nodes,hidden3_nodes]),dtype=tf.float32,name='w3')
        W4 = tf.Variable(var_init([hidden3_nodes,output_feat]),dtype=tf.float32,name='w4')
        
        W_AE = tf.Variable(var_init([hidden3_nodes,input_feat]),dtype=tf.float32,name='w_ae')
    
    with tf.variable_scope('bias'):
        b1 = tf.Variable(tf.zeros(hidden1_nodes),name='b1')
        b2 = tf.Variable(tf.zeros(hidden2_nodes),name='b2')
        b3 = tf.Variable(tf.zeros(hidden3_nodes),name='b3')
        b4 = tf.Variable(tf.zeros(output_feat),name='b4')
        
        B_AE = tf.Variable(tf.zeros(input_feat),name='b_ae')

    parameters = {            
            'W1':W1,
            'W2':W2,
            'W3':W3,
            'W4':W4,
            'W_AE':W_AE,
            'b1':b1,
            'b2':b2,
            'b3':b3,
            'b4':b4,
            'B_AE':B_AE}
    return parameters

def fwd_propagation(x_ph,parameters):
    
    
    W1 = parameters['W1']
    W2 = parameters['W2'] 
    W3 = parameters['W3'] 
    W4 = parameters['W4']  
    W_AE = parameters['W_AE'] 
    
    
    b1 = parameters['b1']  
    b2 = parameters['b2']  
    b3 = parameters['b3']  
    b4 = parameters['b4'] 
    B_AE = parameters['B_AE'] 
    
    act_fn = tf.nn.relu
    
    with tf.variable_scope('layers'):
        hid_layer1 = act_fn(tf.add(tf.matmul(x_ph,W1),b1))
        hid_layer2 = act_fn(tf.add(tf.matmul(hid_layer1,W2),b2))
        hid_layer3 = act_fn(tf.add(tf.matmul(hid_layer2,W3),b3))
    
    with tf.variable_scope('output_layer'):
        output_layer = tf.add(tf.matmul(hid_layer3,W4),b4)
    
    with tf.variable_scope('x_hat_layer'):
        x_hat = tf.nn.sigmoid(tf.add(tf.matmul(hid_layer1,W_AE),B_AE))
    
    return output_layer,x_hat,hid_layer1
    
def fwd_propagation_ae_all(x_ph,parameters):
    ''' all layers of the FC is trained by ae '''
    
    
    W1 = parameters['W1']
    W2 = parameters['W2'] 
    W3 = parameters['W3'] 
    W4 = parameters['W4']  
    W_AE = parameters['W_AE'] 
    
    
    b1 = parameters['b1']  
    b2 = parameters['b2']  
    b3 = parameters['b3']  
    b4 = parameters['b4'] 
    B_AE = parameters['B_AE']  
    
    act_fn = tf.nn.relu
    
    with tf.variable_scope('layer_1'):
        hid_layer1 = act_fn(tf.add(tf.matmul(x_ph,W1),b1))
    with tf.variable_scope('layer_2'):
        hid_layer2 = act_fn(tf.add(tf.matmul(hid_layer1,W2),b2))
    with tf.variable_scope('layer_3'):
        hid_layer3 = act_fn(tf.add(tf.matmul(hid_layer2,W3),b3))
        
    with tf.variable_scope('output_layer'):
        output_layer = tf.add(tf.matmul(hid_layer3,W4),b4)
    
    with tf.variable_scope('x_hat_layer'):
        x_hat = tf.nn.sigmoid(tf.add(tf.matmul(hid_layer3,W_AE),B_AE))
    
    return output_layer,x_hat,hid_layer1,hid_layer2,hid_layer3



