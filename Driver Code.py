from NN_Masking import *
import pandas as pd
from sklearn import preprocessing

'''
X_train = np.genfromtxt("HYStage/X_train_only_cog.csv", delimiter=',')
y_train = np.genfromtxt("HYStage/y_train.csv", delimiter=',')
X_test = np.genfromtxt("HYStage/X_test_only_cog.csv", delimiter=',')
y_test = np.genfromtxt("HYStage/y_test.csv", delimiter=',')
memory_data = [X_train, y_train, X_test, y_test]
'''

'''
train = np.genfromtxt('HYStage/Raw_Dummies_Binary_Train_3908.csv', delimiter=',')
test = np.genfromtxt('HYStage/Raw_Dummies_Binary_Test_1000.csv', delimiter=',')
row, col = train.shape

X_train = train[:,0:(col-1)]
y_train = train[:,(col-1)]
X_test = test[:,0:(col-1)]
y_test = test[:,(col-1)]
memory_data = [X_train, y_train, X_test, y_test]
'''

'''
X = np.genfromtxt('hotel_bookings_cleaned.csv', delimiter=',')
row, col = X.shape
np.random.shuffle(X)
X_train = X[:100000,:(col-1)]
y_train = X[:100000,(col-1)]
X_test = X[100000:,:(col-1)]
y_test = X[100000:,(col-1)]
memory_data = [X_train, y_train, X_test, y_test]
'''

'''
X = np.genfromtxt("C:/Users/Dimitris/Dropbox (UFL)/Machine Learning Masking Code/HYStage_v3/Only_Full_Categorical.csv", delimiter=",")
row, col = X.shape
np.random.shuffle(X)
X_train = X[:1000,:(col-1)]
y_train = X[:1000,(col-1)]
X_test = X[1000:,:(col-1)]
y_test = X[1000:,(col-1)]
memory_data = [X_train, y_train, X_test, y_test]
'''

'''
X = np.genfromtxt("HYStage_v3/Cleaned_Dummies.csv", delimiter=",", skip_header=1)
row, col = X.shape
np.random.shuffle(X)
X_train = X[:10000,:(col-1)]
y_train = X[:10000,(col-1)]
X_test = X[10000:,:(col-1)]
y_test = X[10000:,(col-1)]
memory_data = [X_train, y_train, X_test, y_test]
'''

X_train = np.genfromtxt("Alzheimers/X_train_ADNI.csv", delimiter=",", skip_header=1)
y_train = np.genfromtxt("Alzheimers/y_train_ADNI.csv", delimiter=",", skip_header=1)
X_test = np.genfromtxt("Alzheimers/X_test_ADNI.csv", delimiter=",", skip_header=1)
y_test = np.genfromtxt("Alzheimers/y_test_ADNI.csv", delimiter=",", skip_header=1)

#Min-max the X matrices
min_max_scaler = preprocessing.MinMaxScaler()
X_train = min_max_scaler.fit_transform(X_train)
min_max_scaler = preprocessing.MinMaxScaler()
X_test = min_max_scaler.fit_transform(X_test)

memory_data = [X_train, y_train, X_test, y_test]

neural_network(data=memory_data, masking=True, DP=True, orthogonal=True, category=False, block_size=350, CNN=False, epochs=50, batch_size=10, images=(0, 20, "mnist"), iterations=1)
