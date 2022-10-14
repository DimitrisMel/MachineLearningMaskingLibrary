'''
TODO:
For save_image, expand the indices to cover the -1 case (start from last element), etc.
Save images of other datasets.
'''

import os
import time
import math
import imageio as im
import collections.abc
import numpy as np
from scipy import stats
from keras.datasets import mnist
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Activation
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from sklearn.model_selection import train_test_split

def check_data(data): #Check the type of input and its correctness. Load it or throw exception
    if(isinstance(data, str)):
        if(data.lower() == 'mnist'):
            print("Dataset selected: MNIST")
            #Load mnist
            (X_train, y_train), (X_test, y_test) = mnist.load_data()
            
            #Flatten 28*28 images to a 784 vector for each image
            num_pixels = X_train.shape[1] * X_train.shape[2]
            X_train = X_train.reshape((X_train.shape[0], num_pixels)).astype('float32')
            X_test = X_test.reshape((X_test.shape[0], num_pixels)).astype('float32')
        
            #Normalize inputs from 0-255 to 0-1
            X_train = X_train / 255
            X_test = X_test / 255
        
            #One hot encode outputs
            y_train = np_utils.to_categorical(y_train)
            y_test = np_utils.to_categorical(y_test)
            
            return X_train, y_train, X_test, y_test
    
        elif(data.lower() == 'cifar10'):
            print("Dataset selected: CIFAR10")
            #Load cifar10
            (X_train, y_train), (X_test, y_test) = cifar10.load_data()
            
            #Flatten 32*32*3 images to a 3072 vector for each image
            num_pixels = X_train.shape[1] * X_train.shape[2] * X_train.shape[3]
            X_train = X_train.reshape((X_train.shape[0], num_pixels)).astype('float32')
            X_test = X_test.reshape((X_test.shape[0], num_pixels)).astype('float32')
            
            #Normalize inputs from 0-255 to 0-1
            X_train = X_train / 255
            X_test = X_test / 255
            
            #One hot encode outputs
            y_train = np_utils.to_categorical(y_train)
            y_test = np_utils.to_categorical(y_test)
            
            return X_train, y_train, X_test, y_test
        
        else: #Check for a dataset path
            path = data
            try:
                files = os.listdir(path)
            except:
                raise ImportError("The data directory does not exist.")
            if(all(x in files for x in ["X_train.csv", "y_train.csv", "X_test.csv", "y_test.csv"])): #Check if all data files exist in the directory
                try:
                    X_train = np.genfromtxt(path + "/" + "X_train.csv", delimiter=',')
                    y_train = np.genfromtxt(path + "/" + "y_train.csv", delimiter=',')
                    X_test = np.genfromtxt(path + "/" + "X_test.csv", delimiter=',')
                    y_test = np.genfromtxt(path + "/" + "y_test.csv", delimiter=',')
                except:
                    raise ImportError("Cannot open the files X_train.csv, y_train.csv, X_test.csv, y_test.csv")
                #Handle the case where there is only 1 response vector. Convert the vector into a matrix. Otherwise do one-hot encoding
                if(y_train.ndim == 1):
                    if max(y_train) > 1:
                        y_train = np_utils.to_categorical(y_train)
                        y_test = np_utils.to_categorical(y_test)
                    else:
                        y_train.shape = (-1, 1)
                        y_test.shape = (-1, 1)
            else:
                raise ImportError("Some of the files X_train.csv, y_train.csv, X_test.csv, y_test.csv were not found")
            if(X_train.shape[0] != y_train.shape[0] or X_test.shape[0] != y_test.shape[0] or X_train.shape[1] != X_test.shape[1] or y_train.shape[1] != y_test.shape[1]):
                raise ImportError("The shape of the data files is wrong")
            if(X_train.shape[0] < X_test.shape[0]):
                print("WARNING: The number of rows in X_train is smaller than the number of rows in X_test")
            if(X_train.shape[1] < y_train.shape[1] or X_test.shape[1] < y_test.shape[1]):
                print("WARNING: The number of columns in X_train or X_test is smaller than the number of columns in y_train or y_test")
            print("Custom dataset loaded from path:", path)
            return X_train, y_train, X_test, y_test
    
    elif(isinstance(data, (collections.abc.Sequence, np.ndarray))): #Make sure the data is array like
        try:
            X_train = data[0]
            y_train = data[1]
            X_test = data[2]
            y_test = data[3]
        except:
            ImportError("The data is not in the correct format")
        #Handle the case where there is only 1 response vector. Convert the vector into a matrix. Otherwise do one-hot encoding
        if(y_train.ndim == 1):
            if max(y_train) > 1:
                y_train = np_utils.to_categorical(y_train)
                y_test = np_utils.to_categorical(y_test)
            else:
                y_train.shape = (-1, 1)
                y_test.shape = (-1, 1)
            
        if(X_train.shape[0] != y_train.shape[0] or X_test.shape[0] != y_test.shape[0] or X_train.shape[1] != X_test.shape[1] or y_train.shape[1] != y_test.shape[1]):
            raise ImportError("The shape of the data files is wrong")
        if(X_train.shape[0] < X_test.shape[0]):
            print("WARNING: The number of rows in X_train is smaller than the number of rows in X_test")
        if(X_train.shape[1] < y_train.shape[1] or X_test.shape[1] < y_test.shape[1]):
            print("WARNING: The number of columns in X_train or X_test is smaller than the number of columns in y_train or y_test")
        print("Custom dataset loaded from main memory")
        X_train, y_train, X_test, y_test = reslice(X_train, y_train, X_test, y_test)
        return X_train, y_train, X_test, y_test
    else:
        raise TypeError("The input data must be one of the 4 following options: 1. \"mnist\", 2. \"cifar10\", 3. A path that contains data with names X_train.csv, y_train.csv, X_test.csv, y_test.csv and 4. An array-like in the form [[X_train][y_train][X_test][y_test]]")
        
def reslice(X_train, y_train, X_test, y_test): #Combine the 4 input arrays and then slice them differently each time. Return an array with different versions of slices of these 4 arrays
    X = np.vstack((X_train, X_test))
    y = np.vstack((y_train, y_test))
    X_train_rows = X_train.shape[0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = X_train.shape[0])
    return X_train, y_train, X_test, y_test
        
def check_masking(masking): #Check if the user wants the data to be masked or not
    if(isinstance(masking, bool)):
        if(masking == True):
            print("Masking is enabled")
            return True
        else:
            print("Masking is disabled")
            return False
    else:
        raise TypeError("The argument \"masking\" must be boolean. Set it as True or False")
        
def check_DP(DP, masking): #Check if the user wants to use the DP method (True) or the method without noise addition (False)
    if(DP == None and masking == True): #Default
        print("The DP method is selected by default")
        return True
    elif(DP == None and masking == False):
        return None
    elif(isinstance(DP, bool) and masking == False):
        print("WARNING: You cannot choose the argument \"DP\" without setting masking to \"True\". The input \"DP\" is None")
        return None
    elif(isinstance(DP, bool) and masking == True):
        if(DP == False):
            print("Masking without noise is selected")
            return False
        else:
            print("The DP method is selected")
            return True
    else:
        raise TypeError("The argument \"DP\" must be boolean or None. Please set it as True or False")

def check_orthogonal(orthogonal, masking): #Check if the user wants to use QR factorization (True) or the random method (False)
    if(orthogonal == None and masking == True): #Default
        print("The QR method is selected by default")
        return True
    elif(orthogonal == None and masking == False):
        return None
    elif(isinstance(orthogonal, bool) and masking == False):
        print("WARNING: You cannot choose the argument \"orthogonal\" without setting masking to \"True\". The input \"orthogonal\" is None")
        return None
    elif(isinstance(orthogonal, bool) and masking == True):
        if(orthogonal == False):
            print("Almost orthogonal masking is selected")
            return False
        else:
            print("The QR method is selected")
            return True
    else:
        raise TypeError("The argument \"orthogonal\" must be boolean or None. Please set it as True or False")
        
def check_category(category, masking): #Check if the user wants the data of the same response to be masked together
    if(category == None and masking == True): #Default
        print("Category masking is disabled by default")
        return False
    elif(category == None and masking == False):
        return None
    elif(isinstance(category, bool) and masking == False):
        print("WARNING: You cannot choose the argument \"category\" without setting masking to \"True\". The input \"category\" is None")
        return None
    elif(isinstance(category, bool) and masking == True):
        if(category == True):
            print("Category masking is enabled")
            return True
        else:
            print("Category masking is disabled")
            return False
    else:
        raise TypeError("The argument \"category\" must be boolean or None. Please set it as True or False")
        
def check_block_size(block_size, X_train_rows, orthogonal, masking, num_classes): #The user can set the block size of the mask
    if(block_size == None and masking == True): #Default
        print("The block size of the mask is 1000 by default")
        return 1000
    elif(block_size == None and masking == False):
        return None
    elif(isinstance(block_size, int) and masking == False):
        print("WARNING: You cannot choose the argument \"block_size\" without setting masking to \"True\". The input \"block_size\" is None")
        return None
    elif(isinstance(block_size, int) and masking == True):
        if(block_size <= 1):
            raise ValueError("The block size must be larger than 1")
        if(block_size > X_train_rows):
            raise ValueError("The block size cannot be larger than the rows of X_train")
        if(orthogonal == True and block_size > 5000):
            print("WARNING: The operation will be slow. The block size should be smaller than 5000")
        elif(orthogonal == False and block_size > 50000):
            print("WARNING: The operation will be slow. The block size should be smaller than 50000")
        if(X_train_rows % block_size != 0):
            print("WARNING: The train set is not exactly divisible by the block size. Choose a number that divides", X_train_rows, "exactly")
        print("The block size of the mask is", block_size)
        return block_size
    else:
        raise TypeError("The argument \"block_size\" must be an integer or None")
        
def check_epochs(epochs): #The user can set the number of epochs
    if(epochs == None): #Default
        print("The number of epochs is 10 by default")
        return 10
    elif(isinstance(epochs, int)):
        if(epochs < 1):
            raise ValueError("The number of epochs must be at least 1")
        elif(epochs >= 200):
            print("WARNING: The number of epochs is large. The training operation will be slow")
        print("The number of epochs is", epochs)
        return epochs
    else:
        raise TypeError("The argument \"epochs\" must be an integer or None")
        
def check_batch_size(batch_size): #The user can set the batch size
    if(batch_size == None): #Default
        print("The batch size is 200 by default")
        return 200
    elif(isinstance(batch_size, int)):
        if(batch_size < 1):
            raise ValueError("The batch size must be at least 1")
        elif(batch_size <= 5):
            print("WARNING: The batch size is small. The training operation will be slow")
        print("The batch size is", batch_size)
        return batch_size
    else:
        raise TypeError("The argument \"batch_size\" must be an integer or None")

def check_CNN(CNN): #Check if the user wants to use Feed-Forward (default) or a CNN. Return the bool value or throw exception
    if(CNN == None): #Default
        print("A Feed-Forward Neural Network is selected by default")
        return False
    elif(isinstance(CNN, bool)):
        if(CNN == True):
            print("A Convolutional Neural Network is selected")
            return True
        else:
            print("A Feed-Forward Neural Network is selected")
            return False
    else:
        raise TypeError("The argument \"CNN\" must be boolean or None. Please set it as True or False")

def check_images(images, data, X_train_rows):
    if(images == None):
        print("Saving images is disabled by default")
        return (None, None, None)
    elif(isinstance(data, str)):
        #Check if the types of the tuple of arguments is correct
        if(str(list(map(type, images))[0]) == "<class 'int'>" and str(list(map(type, images))[1]) == "<class 'int'>" and str(list(map(type, images))[2]) == "<class 'str'>"):
            image_start, image_end, image_path = images
            if(image_start >= image_end):
                raise ValueError("The starting index must be smaller than the ending index")
            if(image_start >= X_train_rows or image_end > X_train_rows):
                raise ValueError("The indices must be in the range of the X_train rows, which is 0 -", X_train_rows)
            if(data.lower() == 'mnist'):
                print("Saving images for MNIST is enabled")
                return (image_start, image_end, image_path)
            elif(data.lower() == 'cifar10'):
                print("Saving images for CIFAR10 is enabled")
                return (image_start, image_end, image_path)
            else:
                print("You cannot save images from your own dataset for now. Only MNIST and CIFAR10")
                return (None, None, None)
        else:
            raise TypeError("The argument \"images\" must be of type (int, int, str) for the start_index, end_index and image_path")
    else:
        print("You cannot save images from your own dataset for now. Only MNIST and CIFAR10")
        return (None, None, None)
    
def check_iterations(iterations):
    if(iterations == None): #Default
        print("The number of iterations is 10 by default")
        return 10
    elif(isinstance(iterations, int)):
        if(iterations < 1):
            raise ValueError("The number of iterations must be at least 1")
        elif(iterations >= 100):
            print("WARNING: The number of iterations is too large. The training operation will be slow")
        print("The number of iterations is", iterations)
        return iterations
    else:
        raise TypeError("The argument \"iterations\" must be an integer or None")
    
def QR_generating_and_masking(X, y, DP):
    temp_sum = np.sum(y, axis = 0)
    #only those not equal to 0s are useful
    useful_y =  y[:, np.asarray(np.where(temp_sum > 0.5))[0]]
    del temp_sum
    num_useful = useful_y.shape[1]
    
    row, col = X.shape #Number of rows and columns in the current block (including the response matrix)
    Q1, R1 = np.linalg.qr(np.hstack((useful_y, np.random.normal(size=(row, row - num_useful)))))
    del R1

    Q2, R2 = np.linalg.qr(np.hstack((useful_y, np.random.normal(size=(row, row - num_useful)))))
    del R2
    A = np.matmul(Q1, Q2.T)
    
    if DP == True:
        eps = 0.01
        delta = 0.001
        p = col
        n = row
        gamma_norm = stats.norm.ppf(1-delta)
        bound_a = (gamma_norm*(1+1/(2*gamma_norm**2))/eps)   #equation (11)
        gamma_chi = stats.chi2.ppf(1-delta,df=n*p)
        b = (n-p)*math.sqrt(p)+2*p*gamma_chi
        bound_b = math.sqrt((b+math.sqrt(b**2+8*n*p**2*(n-p)*eps))/(2*(n-p)*eps))   #square root of equation (12)
        sigma = min(bound_a, bound_b)   #equation (13)
        #print("This is SIGMA:", sigma)
        #AXrow, AXcol = AX.shape
        C = np.random.normal(loc=0, scale=sigma, size=(n, p))
        #X = X*1000000
        #print(X)
        #print()
        #print(C)
        #print()
        X = (X + C)
        #print(X)
        
    AX = np.matmul(A, X)
    
    del A, useful_y
    return AX
    
        
def QR_masking(category, X_train, y_train, X_train_rows, X_train_columns, num_classes, block_size, DP):
    if (num_classes == 1):
        y_train = np.hstack((y_train, np.ones((X_train_rows, 1)) - y_train))
        num_classes = 2
    if (category == False):
        num_blocks = X_train_rows // block_size #Floor division to find the number of blocks of size "block_size" each
        Xy = np.hstack((X_train, y_train))
        if(num_blocks >= 1):
            blocks = np.array_split(Xy, num_blocks, axis=0) #This function takes care of non-exact division
        else:
            blocks = [Xy]
        AX_columns = X_train_columns
        AX = np.zeros((0, AX_columns))
        for b in blocks:
            Ab = QR_generating_and_masking(b[:, 0:X_train_columns], b[:, X_train_columns:], DP)
            AX = np.vstack((AX, Ab))
            del Ab
        del blocks, Xy
        return AX
    else:
        AX = np.zeros((0, X_train_columns))
        temp_Xy = np.hstack((X_train, y_train)) 
        order = np.arange(temp_Xy.shape[0])
        pos = 0
        for classes_index  in range(num_classes):
            index_Xy = np.asarray(np.where(temp_Xy[:, X_train_columns + classes_index] > 0.5))[0]
            class_Xy = temp_Xy[index_Xy, :]
            row_num = class_Xy.shape[0]  
            if row_num == 0:
                continue
            for i in range(len(index_Xy)):
                order[index_Xy[i]] = pos + i
            pos += row_num
            AX = np.vstack((AX, QR_masking(False, class_Xy[:, :X_train_columns], class_Xy[:, X_train_columns:], row_num, X_train_columns, num_classes, block_size)))
            del  index_Xy, class_Xy
        del temp_Xy
        AX = AX[order, :]
        return AX
        
    
def random_masking(category, X_train, y_train, X_train_rows, X_train_columns, num_classes, block_size):
    if (num_classes == 1):
        y_train = np.hstack((y_train, np.ones((X_train_rows, 1)) - y_train))
        num_classes = 2
    if(category == False): #Mask the data in the original order
        mask_number = (X_train_rows - 1) // block_size + 1
        first_mask_size = X_train_rows - (mask_number - 1) * block_size
        if first_mask_size == 1 and mask_number > 1:
            first_mask_size += block_size
            mask_number -= 1
        #init AXy
        AXy = np.zeros((0, X_train_columns + y_train.shape[1]))
        pos = 0
        
        #split the training data into batches and mask them
        for i in range(mask_number):
            rownum = block_size
            if i == 0:
                rownum = first_mask_size
            AXy = np.vstack((AXy, random_mask_in_order(X_train[pos : pos + rownum, :], y_train[pos : pos + rownum, :])))
            pos = pos + rownum
        return AXy[:, : X_train_columns]
    else: #Mask the data of each category together
        #init AX
        AX = np.zeros((0, X_train_columns))
        pos = 0
        temp_Xy = np.hstack((X_train, y_train)) 
        order = np.arange(temp_Xy.shape[0])
        for classes_index  in range(num_classes):
            index_Xy = np.asarray(np.where(temp_Xy[:, X_train_columns + classes_index] > 0.5))[0]
            class_Xy = temp_Xy[index_Xy, :]
            row_num = class_Xy.shape[0]  
            if (row_num == 0):
                continue
            for i in range(len(index_Xy)):
                order[index_Xy[i]] = pos + i
            pos += row_num
            AX = np.vstack((AX, random_masking(False, class_Xy[:, :X_train_columns], class_Xy[:, X_train_columns:], row_num, X_train_columns, num_classes, block_size)))
            del class_Xy, index_Xy
        AX = AX[order, :]
        del temp_Xy, order
        return AX
        
def random_mask_in_order(X, y):
    row, col = X.shape
    
    raw_Xy = np.hstack((X, y))
    
    classes_number = y.shape[1]
    
    AXy = np.zeros((row, col + classes_number))
    
    pos = 0
    
    order = np.arange(raw_Xy.shape[0])
    for class_index in range(classes_number):
        index_Xy = np.asarray(np.where(raw_Xy[:, col + class_index] > 0.5))[0]
        class_Xy = raw_Xy[index_Xy , :]
        rownum = class_Xy.shape[0]
        if rownum == 0:
            continue
        for i in range(len(index_Xy)):
            order[index_Xy[i]] = pos + i
        temp_A = np.random.uniform(-1.0, 1.0, (row, rownum - 1))
        #temp_A = np.random.randn(row, rownum - 1)
        temp_A /= np.linalg.norm(temp_A, axis = 0)
        #temp_A /= np.sum(temp_A, axis=0)
        
        index = np.random.randint(0, rownum)    
        
        yr = np.vstack((np.zeros((pos, 1)), np.ones((rownum, 1)), np.zeros((row - pos - rownum, 1))))
        pos = pos + rownum
        y1 = yr - np.matmul(temp_A, np.ones((rownum - 1, 1)))
        
        temp_A = np.hstack((temp_A[:, :index], y1, temp_A[:, index:]))
        
        AXy = AXy + np.matmul(temp_A, class_Xy)
        
        del y1, yr, temp_A, class_Xy, index_Xy
    
    AXy = AXy[order, :]
    del raw_Xy, order
    return AXy

def random_mask_category(X, y):
    row = X.shape[0]
    
    A = np.random.uniform(-1.0, 1.0, (row, row - 1))
    #A = np.random.randn(row, row - 2)
    A /= np.linalg.norm(A, axis = 0)
    #A /= np.sum(A, axis=0)

    y1 = np.ones((row, 1)) - np.matmul(A, np.ones((row - 1, 1)))

    A = np.hstack((A, y1))
    
    AX = np.matmul(A, X)
    del A, y1
    return AX

def create_model_FF(num_classes, X_train_columns):
    model = Sequential()
    model.add(Dense(X_train_columns, input_dim=X_train_columns, activation='relu'))
    if(num_classes == 1):
        model.add(Dense(num_classes, activation='sigmoid'))
        print("Compiling the model...")
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    elif(num_classes > 2):
        model.add(Dense(num_classes, activation='softmax'))
        print("Compiling the model...")
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    else:
        raise ValueError("The number of response vectors cannot be 0 or 2. For binary classification use 1 response vector")
        
    return model

def create_model_CNN(X_train, X_test, num_classes, X_train_rows, image_width, image_height, image_channels):
    # reshape to be [samples][width][height][channels]
    X_test_rows = X_test.shape[0]
    X_train = X_train.reshape((X_train_rows, image_width, image_height, image_channels)).astype('float32')
    X_test = X_test.reshape((X_test_rows, image_width, image_height, image_channels)).astype('float32')
    
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(image_width, image_height, image_channels)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    if(num_classes == 1):
        model.add(Dense(num_classes, activation='sigmoid'))
        print("Compiling the model...")
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    elif(num_classes > 2):
        model.add(Dense(num_classes, activation='softmax'))
        print("Compiling the model...")
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    else:
        raise ValueError("The number of response vectors cannot be 0 or 2. For binary classification use 1 response vector")
    
    return model, X_train, X_test

def create_model_CNN_cifar10(X_train, X_test, num_classes, X_train_rows, image_width, image_height, image_channels):
    # reshape to be [samples][width][height][channels]
    X_test_rows = X_test.shape[0]
    X_train = X_train.reshape((X_train_rows, image_width, image_height, image_channels)).astype('float32')
    X_test = X_test.reshape((X_test_rows, image_width, image_height, image_channels)).astype('float32')
    
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))
    
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))
    
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))
    
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))
    
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    
    print("Compiling the model...")
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model, X_train, X_test

def save_images(X_train, X_train_rows, image_width, image_height, image_channels, image_start, image_end, image_path, masked):
    X_train = X_train.reshape((X_train_rows, image_width, image_height, image_channels)).astype('float32')
    if not os.path.exists(image_path):
        try:
            os.makedirs(image_path)
        except:
            print("Wrong image directory")
    if(masked == False):
        for i in range(image_start, image_end):
            im.imwrite((image_path + "/" + "%s_raw.png" %i), (X_train[i,:]*256).astype(np.uint8))
    else:
        for i in range(image_start, image_end):
            im.imwrite((image_path + "/" + "%s_masked.png" %i), (X_train[i,:]*256).astype(np.uint8))
            
def training(model, X_train, y_train, X_test, y_test, epochs, batch_size):
    print("Training the model...")
    model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs=epochs, batch_size=batch_size) #Fit the model
    
    scores = model.evaluate(X_test, y_test, verbose=0) #Final evaluation of the model
    print("Accuracy: %.2f%%" % (scores[1]*100))
            
def neural_network(data, masking, DP = None, orthogonal = None, category = None, block_size = None, CNN = None, epochs = None, batch_size = None, images = None, iterations = None):
    start_time = time.time()
    #Check all inputs
    iterations = check_iterations(iterations)
    for i in range(iterations):
        X_train, y_train, X_test, y_test = check_data(data)
        if(isinstance(data, str) and data.lower() == 'mnist'):
            image_width, image_height, image_channels = 28, 28, 1
        elif(isinstance(data, str) and data.lower() == 'cifar10'):
            image_width, image_height, image_channels = 32, 32, 3
        X_train_rows, X_train_columns = X_train.shape
        y_test_rows, y_test_columns = y_test.shape
        num_classes = y_test_columns
        num_pixels = X_train_columns
        print("Data has been loaded. Total time elapsed: %s seconds" % (time.time() - start_time))
        masking = check_masking(masking)
        DP = check_DP(DP, masking) #Only applicable to the QR method, not the random method
        orthogonal = check_orthogonal(orthogonal, masking)
        category = check_category(category, masking)
        block_size = check_block_size(block_size, X_train_rows, orthogonal, masking, num_classes)
        CNN = check_CNN(CNN)
        epochs = check_epochs(epochs)
        batch_size = check_batch_size(batch_size)
        image_start, image_end, image_path = check_images(images, data, X_train_rows)

        '''
        #Debugging block
        print("***Variables***")
        print("X_train:", X_train)
        print("y_train:", y_train)
        print("X_test:", X_test)
        print("y_test:", y_test)
        print("X_train_rows, X_train_columns:", X_train_rows, X_train_columns)
        print("y_test_rows, y_test_columns:", y_test_rows, y_test_columns)
        print("masking:", masking)
        print("category:", category)
        print("orthogonal:", orthogonal)
        print("block_size:", block_size)
        print("CNN:", CNN)
        print("image_start, image_end, image_path, image_width, image_height, image_channels:", image_start, image_end, image_path, image_width, image_height, image_channels)
        print("num_classes:", num_classes)
        print("num_pixels:", num_pixels)
        '''

        if(image_path != None): #Save raw images
            save_images(X_train, X_train_rows, image_width, image_height, image_channels, image_start, image_end, image_path, False)

        if(masking == True and orthogonal == True): #Mask the data with QR created mask
            print("Creating masking matrix...")
            X_train = QR_masking(category, X_train, y_train, X_train_rows, X_train_columns, num_classes, block_size, DP)
            print("The data has been masked. Total time elapsed: %s seconds" % (time.time() - start_time))
        elif(masking == True and orthogonal == False): #Mask the data with the random masking method
            print("Creating masking matrix...")
            X_train = random_masking(category, X_train, y_train, X_train_rows, X_train_columns, num_classes, block_size)
            print("The data has been masked. Total time elapsed: %s seconds" % (time.time() - start_time))

        if(image_path != None and masking == True): #Save masked images
            save_images(X_train, X_train_rows, image_width, image_height, image_channels, image_start, image_end, image_path, masking)
        
        if(CNN == False):
            model = create_model_FF(num_classes, X_train_columns)
        else: #If CNN is True, use the CNN instead of FF
            if(isinstance(data, str) and data.lower() == 'cifar10'):
                model, X_train, X_test = create_model_CNN_cifar10(X_train, X_test, num_classes, X_train_rows, image_width, image_height, image_channels)
            else:
                model, X_train, X_test = create_model_CNN(X_train, X_test, num_classes, X_train_rows, image_width, image_height, image_channels)
    
        training(model, X_train, y_train, X_test, y_test, epochs, batch_size)
    
        print("The model has been trained and evaluated. Total time elapsed: %s seconds" % (time.time() - start_time))

        del X_train, y_train, X_test, y_test, model