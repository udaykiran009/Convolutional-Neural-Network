import numpy as np
import gzip

# x = np.random.random((2,8,8)) - 0.5


def ReLU(x):
    """
    param:
        x : numpy array object
    returns:
        modified x with ReLU function aaplied 
    """
    x[x<0] = 0
    return x

def nanargmax(a):
    """
    param: 
        a : numpy array object for which we find the index at which max occurs
    returns:
        index of the max value that occurs
    """
    idx = np.argmax(a, axis=None)
    max_indx = np.unravel_index(idx, a.shape)
    if np.isnan(a[max_indx]):
        nan_count = np.sum(np.isnan(a))
        # In numpy < 1.8 use idx = np.argsort(a, axis=None)[-nan_count-1]
        idx = np.argpartition(a, -nan_count-1, axis=None)[-nan_count-1]
        max_indx = np.unravel_index(idx, a.shape)
    return max_indx

# nanargmax return the indices of the maximum value in that array
def maxpool(X, f, s):
    """
    param:
        X : numpy object array
        f : the size of the kernel
        s : stride for maxpooling
    returns:
        maxpooled output of matrix X
        
    """
    (l, w, w) = X.shape
    pool = np.zeros((l, (w-f)/s+1,(w-f)/s+1))
    for jj in range(0,l):
        i=0
        while(i<w):
            j=0
            while(j<w):
                pool[jj,i/2,j/2] = np.max(X[jj,i:i+f,j:j+f])
                j+=s
            i+=s
    return pool

# def maxpool(X,w,s):
#   # size*size is image size
#   # Image must be square shaped
#   channels,size,size = X.shape
#   pool = np.zeros((channels, int(np.ceil(size/(w+s-1))),int(np.ceil(size/(w+s-1))) ))
#   pool_size = pool.shape[1]
#   # print(pool.shape)
#   for channel in range(channels):
#       ip=0
#       i = 0
#       j = 0
#       while(ip<pool_size):
#           jp=0
#           j = 0
#           while(jp<pool_size):
#               pool[channel,ip,jp]  = np.max(X[channel,i:i+w,j:j+w])
#               j += w + s -1
#               jp = jp+1
#           i += w + s - 1
#           ip = ip+1
#   # print(pool)

#   return pool



def softmax_cost(out,y):

    """
    param : 
        out : output of the fully connected layer
        y : output label 10x1 matrix

    returns : 
        costs : cost because of the classification
        probability : probability of belonging to any of the classes
    """
    eout = np.exp(out, dtype=np.float) #we dont have 128 a typo fixed
    probability = eout/sum(eout)
    
    p = sum(y*probability) #y is the output label [0 0 0 1 0 0 ... ]

    cost = -np.log(p)   ## (Only data loss. No regularised loss)
    return cost,probability 


       #Returns gradient for all the paramaters in each iteration
def ConvuluteNetwork(image, label, kernel1, kernel2, biasconv1, biasconv2, weights, biasfc):


    """
    param : 
        image : input image of size 28x28x1
        kernel1 : conolutional layer 1 kernels
        kernel2 : convolutional layer 2 kernels
        biasconv1 : bias for layer 1
        biasconv2 : bias for layer 2
        weights : weight matrix for fully connected layer 800x10
        biasfc : bias for fully connected layer
    

    returns : 
        dkernel1 : changes to be added to kernel1
        dkernel2 : changes to be added to kernel2
        dbiasconv1 : changes to be added to biasconv1
        dbiasconv2 : changes to be added to biasconv2
        dweights : changes to be added to weights
        dbiasfc : changes to be added to biasfc 
        cost : calculates cost using softmax function 
        accuracy : accuracy is accuracy

    """
    # feed forward building the network
    # calculations for conv 1 layer

    # l - channel
    # w - size of square image

    (l, w, w) = image.shape 

    # l1 - No. of filters in Conv1
    # l2 - No. of filters in Conv2

    l1 = len(kernel1)
    l2 = len(kernel2)
    ( _, f, f) = kernel1[0].shape

    # w1 - size of image after convolute1
    # w2 - size of image after convolute2

    w1 = w-f+1
    w2 = w1-f+1

    
    convolute1 = np.zeros((l1,w1,w1))
    convolute2 = np.zeros((l2,w2,w2))  


    for jj in range(0,l1):
        for x in range(0,w1):
            for y in range(0,w1):

                convolute1[jj,x,y] = np.sum( image[:,x:x+f,y:y+f] * kernel1[jj] ) + biasconv1[jj]

    ReLU(convolute1) 

    #calculations of the 2nd conv layer starts from here

    for jj in range(0,l2):
        for x in range(0,w2):
            for y in range(0,w2):

                convolute2[jj,x,y] = np.sum( convolute1[:,x:x+f,y:y+f] * kernel2[jj] ) + biasconv2[jj]

    ReLU(convolute2) #relu activation

    # maxpooling with window size 2 and stride 2

    pooled_layer = maxpool(convolute2, 2, 2)    

    fully_connected = pooled_layer.reshape((int((w2/2))*int((w2/2))*l2,1)) #starting point of fully connected layer
    
    out = weights.dot(fully_connected) + biasfc #output of size 10x1
    
    #Softmax function for probabilities
    
    cost, probability = softmax_cost(out, label)
    if np.argmax(out)==np.argmax(label):
        acc=1
    else:
        acc=0

    
    # Backpropagation for getting the differentiation using chain rule
    
    dout = probability - label  #dL/dout  10x1
    
    dweights = dout.dot(fully_connected.T)  #dL/dweights

    dbiasfc = sum(dout.T).T.reshape((10,1))     #dbiasfc    

    dfully_connected = weights.T.dot(dout)      #dL/dfully_connected

    dpool = dfully_connected.T.reshape((l2, int((w2/2)), int((w2/2))))

    dconvolute2 = np.zeros((l2, w2, w2))
    
    for jj in range(0,l2):
        i=0
        while(i<w2):
            j=0
            while(j<w2):
                (a,b) = nanargmax(convolute2[jj,i:i+2,j:j+2]) #Getting indices of maximum value in the array
                dconvolute2[jj,i+a,j+b] = dpool[jj,int(i/2),int(j/2)]
                j+=2
            i+=2
    
    dconvolute2[convolute2<=0]=0 #ReLU

    dconvolute1 = np.zeros((l1, w1, w1))
    dkernel2 = {}
    dbiasconv2 = {}
    for xx in range(0,l2):
        dkernel2[xx] = np.zeros((l1,f,f))
        dbiasconv2[xx] = 0

    dkernel1 = {}
    dbiasconv1 = {}
    for xx in range(0,l1):
        dkernel1[xx] = np.zeros((l,f,f))
        dbiasconv1[xx] = 0

    for jj in range(0,l2):
        for x in range(0,w2):
            for y in range(0,w2):
                dkernel2[jj]+=dconvolute2[jj,x,y]*convolute1[:,x:x+f,y:y+f]
                dconvolute1[:,x:x+f,y:y+f]+=dconvolute2[jj,x,y]*kernel2[jj]
        dbiasconv2[jj] = np.sum(dconvolute2[jj])
    dconvolute1[convolute1<=0]=0
    for jj in range(0,l1):
        for x in range(0,w1):
            for y in range(0,w1):
                dkernel1[jj]+=dconvolute1[jj,x,y]*image[:,x:x+f,y:y+f]

        dbiasconv1[jj] = np.sum(dconvolute1[jj])

    
    return [dkernel1, dkernel2, dbiasconv1, dbiasconv2, dweights, dbiasfc, cost, acc]




def initialize_theta(nout, nin):

    """
    param : 
        nout : output neurons
        nin : input neurons

    returns : 
         returns a matrix of size (in X out) with randomly initialized weights

    """
    return 0.01*np.random.rand(nout, nin)

def init_kernel(kernel_dim, n_channels, distribution='normal'):

    """
    param : 
        kernel_dim : size of the window
        n_channels : number of channels. 1 here

    returns : 
         a kernel with values sampled from gaussian distribution

    """
    
    #defining standard deviation for the normal distribution of the kernel initialization
    std_shape = kernel_dim*kernel_dim*n_channels    
    stddev = np.sqrt(1./std_shape)
    kernel_shape = (n_channels,kernel_dim,kernel_dim) # kernel shape
    return np.random.normal(loc = 0,scale = stddev,size = kernel_shape) 

## Returns all the trained parameters
def Momentum_Grad_des(batch, learningrate, w, l, MU, kernel1, kernel2, biasconv1, biasconv2, weights, biasfc, cost, acc):

    """
    param : 
        batch : batch for claculating gradient descent
        learningrate : the rate at which the changes are updated 
        w : size of the image
        l : channel length
        MU : momentum
        kernel1 : conolutional layer 1 kernels
        kernel2 : convolutional layer 2 kernels
        biasconv1 : bias for layer 1
        biasconv2 : bias for layer 2
        weights : weight matrix for fully connected layer 800x10
        biasfc : bias for fully connected layer 
        cost : calculates cost using softmax function 
        accuracy : accuracy  

    returns : 
        kernel1 : updated kernel1 after gradient descent 
        kernel2 : updated kernel2 after gradient descent  
        biasconv1 : updated biasconv1 after gradient descent
        biasconv2 : updated biasconv2 after gradient descent 
        weights : updated weights after gradient descent 
        biasfc : updated biasfc after gradient descent 
        cost : updated cost after gradient descent 
        accuracy : updated accuracy after gradient descent

    """
        
    X = batch[:,0:-1]
    X = X.reshape(len(batch), l, w, w)
    y = batch[:,-1]

    n_correct=0
    cost_ = 0
    batch_size = len(batch)
    dkernel2 = {}
    dkernel1 = {}
    dbiasconv2 = {}
    dbiasconv1 = {}
    v1 = {}
    v2 = {}
    bv1 = {}
    bv2 = {}
    for k in range(0,len(kernel2)):
        dkernel2[k] = np.zeros(kernel2[0].shape)
        dbiasconv2[k] = 0
        v2[k] = np.zeros(kernel2[0].shape)
        bv2[k] = 0
    for k in range(0,len(kernel1)):
        dkernel1[k] = np.zeros(kernel1[0].shape)
        dbiasconv1[k] = 0
        v1[k] = np.zeros(kernel1[0].shape)
        bv1[k] = 0
    dweights = np.zeros(weights.shape)
    dbiasfc = np.zeros(biasfc.shape)
    v3 = np.zeros(weights.shape)
    bv3 = np.zeros(biasfc.shape)



    for i in range(0,batch_size):
        
        image = X[i]

        label = np.zeros((weights.shape[0],1))
        label[int(y[i]),0] = 1
        
        ## Fetching gradient for the current parameters
        [dkernel1_, dkernel2_, dbiasconv1_, dbiasconv2_, dweights_, dbiasfc_, curr_cost, acc_] = ConvuluteNetwork(image, label, kernel1, kernel2, biasconv1, biasconv2, weights, biasfc)
        for j in range(0,len(kernel2)):
            dkernel2[j]+=dkernel2_[j]
            dbiasconv2[j]+=dbiasconv2_[j]
        for j in range(0,len(kernel1)):
            dkernel1[j]+=dkernel1_[j]
            dbiasconv1[j]+=dbiasconv1_[j]
        dweights+=dweights_
        dbiasfc+=dbiasfc_

        cost_+=curr_cost
        n_correct+=acc_

    for j in range(0,len(kernel1)):
        v1[j] = MU*v1[j] -learningrate*dkernel1[j]/batch_size
        kernel1[j] += v1[j]
        # kernel1[j] -= learningrate*dkernel1[j]/batch_size
        bv1[j] = MU*bv1[j] -learningrate*dbiasconv1[j]/batch_size
        biasconv1[j] += bv1[j]
    for j in range(0,len(kernel2)):
        v2[j] = MU*v2[j] -learningrate*dkernel2[j]/batch_size
        kernel2[j] += v2[j]
        # kernel2[j] += -learningrate*dkernel2[j]/batch_size
        bv2[j] = MU*bv2[j] -learningrate*dbiasconv2[j]/batch_size
        biasconv2[j] += bv2[j]
    v3 = MU*v3 - learningrate*dweights/batch_size
    weights += v3
    # weights += -learningrate*dweights/batch_size
    bv3 = MU*bv3 -learningrate*dbiasfc/batch_size
    biasfc += bv3

    cost_ = cost_/batch_size
    cost.append(cost_)
    accuracy = float(n_correct)/batch_size
    acc.append(accuracy)

    return [kernel1, kernel2, biasconv1, biasconv2, weights, biasfc, cost, acc]

## Predict class of each row of matrix X
def predict(image, kernel1, kernel2, biasconv1, biasconv2, weights, biasfc):

    """
    param : 
        image : input image 
        filt : final kernel for convolute1 used to classify the image 
        kernel2 : final kernel for convolute2 used to classify the image
        biasconv1 : final biasconv1 used to classify the image 
        biasconv2 : final biasconv2 used to classify the image 
        weights : final weights for fc layer
        bias : final bias for fully connected layer

    returns : 
         number of maximum probability
         probability of that image belonging to that class
        
    """

    (l,w,w)=image.shape
    (l1,f,f) = kernel2[0].shape
    l2 = len(kernel2)
    w1 = w-f+1
    w2 = w1-f+1
    convolute1 = np.zeros((l1,w1,w1))
    convolute2 = np.zeros((l2,w2,w2))
    for jj in range(0,l1):
        for x in range(0,w1):
            for y in range(0,w1):
                convolute1[jj,x,y] = np.sum(image[:,x:x+f,y:y+f]*kernel1[jj])+biasconv1[jj]
    convolute1[convolute1<=0] = 0 #relu activation

    #Calculating second Convolution layer
    for jj in range(0,l2):
        for x in range(0,w2):
            for y in range(0,w2):
                convolute2[jj,x,y] = np.sum(convolute1[:,x:x+f,y:y+f]*kernel2[jj])+biasconv2[jj]
    convolute2[convolute2<=0] = 0 #relu activation

    # maxpooling with a kernel size 2 and stride 2
    pooled_layer = maxpool(convolute2, 2, 2)    
    # print(w2)
    fully_connected = pooled_layer.reshape(((w2/2)*(w2/2)*l2,1))
    #print(weights.size, fully_connected.size, biasfc.size)
    out = weights.dot(fully_connected) + biasfc #10*1
    eout = np.exp(out, dtype=np.float)
    probability = eout/sum(eout)
    # probability = 1/(1+np.exp(-out))

    # print out
    # print np.argmax(out), np.max(out)
    return np.argmax(probability), np.max(probability)


def extract_data(filename, num_images, IMAGE_WIDTH):

# this function definition has been taken from internet


    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_WIDTH * IMAGE_WIDTH * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32) #Interpret a buffer as a 1-dimensional array
        data = data.reshape(num_images, IMAGE_WIDTH*IMAGE_WIDTH)
        return data

def extract_labels(filename, num_images):

# this function definition has been taken from internet

    """Extract the labels into a vector of int64 label IDs.""" 
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64) #Interpret a buffer as a 1-dimensional array
    return labels
