import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from matplotlib import gridspec
import pickle
import time
import random
from CNN_layer import *
from remaining_time import *




#Hyperparameters
Output_number = 10
learningrate = 0.01	#learning rate
Image_size = 28
Image_channel = 1
kernl_shape=5
kernl1_number = 8
kernl2_number =8
batch_sze = 20
epoch_number = 2	 # number of iterations
mu = 0.98

#PICKLE_FILE = 'output.pickle'
PICKLE_FILE = 'trained.pickle'


## Data extracting
m =10000
X = extract_data('t10k-images-idx3-ubyte.gz', m, Image_size)
y_dash = extract_labels('t10k-labels-idx1-ubyte.gz', m).reshape(m,1)
image=X[99]
plt.imshow(image.reshape((28,28)))
plt.show()

# print(X.shape)
X-= int(np.mean(X))
X/= int(np.std(X))
print(X)
image=X[99]
plt.imshow(image.reshape((28,28)))
plt.show()
test_data = np.hstack((X,y_dash))

# print(test_data)

# print('loaded00000000000000000000000000000000000000')

m =50000
X = extract_data('train-images-idx3-ubyte.gz', m, Image_size)
y_dash = extract_labels('train-labels-idx1-ubyte.gz', m).reshape(m,1)
# print (np.mean(X), np.std(X))
X-= int(np.mean(X)) #mean is set to 0

X/= int(np.std(X)) #data is set is normalized to 11
print(np.std(X) , np.mean(X))

train_data = np.hstack((X,y_dash))

np.random.shuffle(train_data)



digi_img = train_data.shape[0]
print(digi_img, "yahoo")

## Initializing all the parameters
kernel1 = {}
kernel2 = {}
biasconv1 = {}
biasconv2 = {}
for i in range(0,kernl1_number):
	kernel1[i] = init_kernel(kernl_shape, Image_channel, distribution='normal')
	biasconv1[i] = 0
	# v1[i] = 0
for i in range(0,kernl2_number):
	kernel2[i] = init_kernel(kernl_shape, kernl1_number, distribution='normal')
	biasconv2[i] = 0
	# v2[i] = 0
w1 = Image_size-kernl_shape+1
w2 = w1-kernl_shape+1
theta3 = initialize_theta(Output_number, int((w2/2))*int((w2/2))*kernl2_number)

biasfc = np.zeros((Output_number,1))
cost = []
accuracy = []
# pickle_in = open(PICKLE_FILE, 'rb')
# out = pickle.load(pickle_in)

# [kernel1, kernel2, biasconv1, biasconv2, theta3, biasfc, cost, accuracy] = out

xrange = range

print("Learning Rate:"+str(learningrate)+", Batch Size:"+str(batch_sze))

## Training start here

for epoch in range(0,epoch_number):
	np.random.shuffle(train_data)
	#xrange gives 20 if number of train images is less than 20 or the number of train images if higher
	batches = [train_data[k:k + batch_sze] for k in xrange(0, digi_img, batch_sze)] 
	x=0
	for batch in batches:
		#time is started for that batch
		stime = time.time() 

		# learningrate =  learningrate/(1+epoch/10.0)
		out = Momentum_Grad_des(batch, learningrate, Image_size, Image_channel, mu, kernel1, kernel2, biasconv1, biasconv2, theta3, biasfc, cost, accuracy)
		[kernel1, kernel2, biasconv1, biasconv2, theta3, biasfc, cost, accuracy] = out
		print(kernel1[0].shape, kernel2[0].shape, "yesss")
		epoch_accuracy = round(np.sum(accuracy[int(epoch*digi_img/batch_sze):])/(x+1),2)
		
		per = float(x+1)/len(batches)*100
		print("Epoch:"+str(round(per,2))+"% Of "+str(epoch+1)+"/"+str(epoch_number)+", Cost:"+str(cost[-1])+", B.Acc:"+str(accuracy[-1]*100)+", E.Acc:"+str(epoch_accuracy))
		
		ftime = time.time()
		deltime = ftime-stime
		remtime = (len(batches)-x-1)*deltime+deltime*len(batches)*(epoch_number-epoch-1)
		printTime(remtime)
		x+=1


## saving the trained model parameters
with open(PICKLE_FILE, 'wb') as file:
	pickle.dump(out, file)

## Opening the saved model parameter
pickle_in = open(PICKLE_FILE, 'rb')
out = pickle.load(pickle_in)

[kernel1, kernel2, biasconv1, biasconv2, theta3, biasfc, cost, accuracy] = out


## Plotting the cost and accuracy over different background
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1]) 
ax0 = plt.subplot(gs[0])
line0, = ax0.plot(cost, color='b')
ax1 = plt.subplot(gs[1], sharex = ax0)
line1, = ax1.plot(accuracy, color='r', linestyle='--')
plt.setp(ax0.get_xticklabels(), visible=False)
ax0.legend((line0, line1), ('Loss', 'Accuracy'), loc='upper right')
# remove vertical gap between subplots
plt.subplots_adjust(hspace=.0)
plt.show(block=False)

## Computing Test accuracy
X = test_data[:,0:-1]
X = X.reshape(len(test_data), Image_channel, Image_size, Image_size)
y = test_data[:,-1]
print('dhjadgfhdsbfk')
print(y.shape)
corr = 0
print("Computing accuracy over test set:")
for i in range(0,len(test_data)):
	image = X[i]
	digit, prob = predict(image, kernel1, kernel2, biasconv1, biasconv2, theta3, biasfc)
	print (digit, y[i], prob)
	if digit==y[i]:
		corr+=1
	if (i+1)%int(0.01*len(test_data))==0:
		print(str(float(i+1)/len(test_data)*100)+"% Completed")
test_accuracy = float(corr)/len(test_data)*100
print("Test Set Accuracy:"+str(test_accuracy))
