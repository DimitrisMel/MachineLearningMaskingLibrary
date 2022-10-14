# MachineLearningMaskingLibrary
This is a Python library for Matrix Masking in Machine Learning applications.

This software has been tested on python 3.8.5, CUDA 10.1, CuDNN 7.6.5 and tensorflow 2.3.0 on NVidia GPUs series 10xx and 20xx.

The package dependencies are: imageio, numpy, keras and tensorflow


# Function call

neural_network(data, masking, orthogonal, category, block_size, CNN, epochs, batch_size, images, iterations)

Train and evaluate a neural network with raw or masked data.

Parameters
data: str or array-like, required
If data is “mnist”, then the MNIST dataset it downloaded. If data is “cifar10”, then the CIFAR10 dataset is downloaded. If the data is str but not mnist or cifar10, then it is considered a path to get the data files from which must contain the files X_train.csv, y_train.csv, X_test.csv, y_test.csv. If the path of the files is the same as the path of the program, use the absolute path like data=”C:\Users\Username\ML Matrix Masking”. Finally, the data can be a python array or numpy array in the main memory, in the format of [[X_train][y_train][X_test][y_test]], where each of the 4 components is a 2D array (rows are records and columns are attributes).

masking: bool, required
If masking is True, then the data is masked before it trains the neural network. If masking is False, the raw data is used for training.

DP: bool, optional, default=True
If DP is True, then the Differential Privacy method is used to produce the masking matrix. If DP is False, then there is no noise added to the masking. This parameter is only considered when masking is True. If none is given by the user, it’s set to True by default.

orthogonal: bool, optional, default=True
If orthogonal is True, then the QR factorization method is used to produce an orthogonal masking matrix. If orthogonal is False, then the random weight method is used to produce an almost orthogonal masking matrix. This parameter is only considered when masking is True. If none is given by the user, it’s set to True by default.

category: bool, optional, default=False
If category is True, then the data is masked according their response category. The rows of the same response are masked together. If category is False, then the data is masked in the original order. This parameter is only considered when masking is True. If none is given by the user, it’s set to False by default.


block_size: int, optional, default=1000
If the number of rows in X_train is larger than block_size, then the data will be masked in groups of size “block_size”. This parameter is only considered when masking is True. If none is given by the user, it’s set to 1000 by default. block_size must be > 1.

CNN: bool, optional, default=False
If CNN is True then the data is trained through a Convolutional Neural Network. If CNN is False, then a Feed-Forward neural network is used. If none is given by the user, it’s set to False by default.
Note that the CIFAR10 dataset uses a deeper CNN than MNIST.

epochs: int, optional, default=10
	Specify how many times to run through the X_train. If none is given, it’s set to 10 by default.

batch_size: int, optional, default=200
	Specify the size of each training batch. If none is given, it’s set to 200 by default.

images: object, optional, default=None
Parameter that the user can set in order to save images from specific indices of X_train. It is a tuple of (int, int, str) for the start_index, end_index and image_path. The index_path is used to determine the sub-directory to save the images into. This works only for MNIST and CIFAR10.

iterations: int, optional, default=10
	Specify the number of runs of training and validation. Each time the data is combined and resliced.



Returns
None

Example 1: neural_network(data=”mnist”, masking=True, DP=True, orthogonal=False, category=True, block_size=100, CNN=True, epochs=20, batch_size=10, images=(10, 20, ”images/mnist”), iterations=10)
Example 2: neural_network(data=”cifar10”, masking= False, CNN=True, epochs=350, batch_size=500, images=(10, 20, ”images/cifar10”), iterations=20)


Dependencies
imageio
numpy
keras
tensorflow
scipy

Also, the python native dependencies needed are:
os
collections
time
math
