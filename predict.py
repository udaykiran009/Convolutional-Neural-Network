import pickle
from CNN_layer import *



## Hyperparameters
Image_size = 28
Image_channel = 1
PICKLE_FILE = 'trained.pickle'

## Data extracting
m =10000
X = extract_data('t10k-images-idx3-ubyte.gz', m, Image_size)
y = extract_labels('t10k-labels-idx1-ubyte.gz', m).reshape(m,1)
X-= int(np.mean(X))
X/= int(np.std(X))


## Opening the saved model parameter
pickle_in = open(PICKLE_FILE, 'rb')
out = pickle.load(pickle_in)
[kernel1, kernel2, biasconv1, biasconv2, theta3, biasfc, cost, accuracy] = out


for i in range(20,50):
	image = X[i].reshape(Image_channel, Image_size, Image_size)
	digit, prob = predict(image, kernel1, kernel2, biasconv1, biasconv2, theta3, biasfc)
	print digit, prob, y[i][0]
