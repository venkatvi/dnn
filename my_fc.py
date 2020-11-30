#Quantization Algorithm for convolution layer

import numpy as np
import math 
def computeExponent(data, WL):
	dataVec = data.flatten();
	maxVal = np.max(np.abs(dataVec))
	output = math.frexp(maxVal);
	output = np.asarray(output); 
	exponent = output[1]; 
	#exponent-=1;
	return exponent - (WL - 1); 

def scale(X, scalingFactor):
	if scalingFactor == 0:
		return X;
	Y = X * 2**(-1*scalingFactor)
	Y[Y > 127] = 127; 
	Y[Y < -128] = -128;
	Y = np.round(Y);
	Y = Y.astype(np.int8) 
	return Y;

def scaleToInt32(X, scalingFactor):
	if scaleToInt32 == 0:
		return X;
	Y = X * 2**(-1 * scalingFactor)
	Y[Y > 2**31-1] = 2**32-1;
	Y[Y < -2^31] = -2^31;
	Y = np.round(Y);
	Y = Y.astype(np.int); 
	return Y;


def rescale(Y, rescaleFactor):
	if (rescaleFactor == 0):
		return Y;
	Y = Y.astype(np.float) * (2**np.float(rescaleFactor)); 
	return Y;
def computeOutputSize(X, W):
	X = np.array(X);
	W = np.array(W); 
	
	inputSize = np.asarray(X.shape)
	inputHW = inputSize[0:2]
	
	filterSize = np.asarray(W.shape)
	filterHW = filterSize[0:2]
	

	
	outputSize = np.array([1, 1, filterSize[0], inputSize[3]]);
	outputSize = outputSize.astype(np.int)
	
	return outputSize; 

def fc(X, weight, bias, exponentsDict):
	outputSize = computeOutputSize(X, weight)
	
	output = np.zeros(outputSize, dtype=np.float);
	outputSize = np.asarray(output.shape)
	
	filterSize = np.asarray(weight.shape)
	numFilters = filterSize[1];
	X = np.reshape(X, (numFilters, -1));

	if (exponentsDict['input'] != 0):
		qX = scale(X, exponentsDict['input'])
		qW = scale(weight, exponentsDict['weight'])
		qB = scaleToInt32(bias, exponentsDict['input'] + exponentsDict['weight'])
		output = output.astype(np.int)
		output = np.matmul(qW.astype(np.int), qX.astype(np.int)) + qB;
		output = rescale(output, exponentsDict['input'] + exponentsDict['weight'])
	else:
		output = np.matmul(weight, X) + bias;
			
	
	print(output.shape)
	return output;

if __name__ == "__main__":
	X = np.random.randint(-32, 31, (1, 1, 64, 1)); 
	W = np.random.normal(0, 1, (1000, 64))
	b = np.ones((1000, 1))

	stride = 2
	padding = np.ones(4)
	dilation = 1
	
	exponentsDict = {'input': 0, 'weight': 0, 'bias': 0};

	floatConv = fc(X, W, b, exponentsDict)
	floatConv = floatConv.flatten();

	exponentsDict['input'] = computeExponent(X, 8);
	exponentsDict['weight'] = computeExponent(W, 8);

	print(exponentsDict)

	int8Conv = fc(X, W, b, exponentsDict)
	int8Conv = int8Conv.flatten();


	qX = scale(X, exponentsDict['input']);
	rescaledX = rescale(qX, exponentsDict['input']); 
	diff = np.sum((X-rescaledX)**2)/X.size;
	print(diff)

	qW = scale(W, exponentsDict['weight']);
	rescaledW = rescale(qW, exponentsDict['weight']); 
	diff = np.sum((W-rescaledW)**2)/W.size;
	print(diff)

	qB = scaleToInt32(b, exponentsDict['input'] + exponentsDict['weight'])
	rescaledB = rescale(qB, exponentsDict['input'] + exponentsDict['weight']); 
	diff = np.sum((b-rescaledB)**2)/b.size;
	print(diff)

	diff = np.sum((floatConv-int8Conv)**2)/floatConv.size 
	print(diff)



