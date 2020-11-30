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
def computeOutputSize(X, W, paddingSize, strideHW, dilationHW):
	X = np.array(X);
	W = np.array(W); 
	paddingSize = np.array(paddingSize)

	inputSize = np.asarray(X.shape)
	inputHW = inputSize[0:2]
	
	filterSize = np.asarray(W.shape)
	filterHW = filterSize[0:2]
	filterHW = np.multiply(dilationHW, (filterHW-1)) + 1; 
	

	top = 0; bottom = 1; left = 2; right = 3; 
	paddingHW = np.asarray((paddingSize[top] + paddingSize[bottom], paddingSize[left] + paddingSize[right])); 
	
	a1 = np.add(inputHW, paddingHW)
	a2 = -1 * filterHW
	a3 = np.add(a1, a2)

	num = a3.astype(np.float)
	
	dem = np.float(1/np.float(strideHW))
	
	outputHW = np.floor(np.multiply( num, dem )) + 1; 
	
	outputSize = np.array([outputHW[0], outputHW[1], filterSize[3], inputSize[3]]);
	outputSize = outputSize.astype(np.int)
	
	return outputSize; 

def conv(X, weight, bias, stride, padding, dilation, exponentsDict):
	outputSize = computeOutputSize(X, weight, padding, stride, dilation)
	
	output = np.zeros(outputSize, dtype=np.float);
	outputSize = np.asarray(output.shape)
	
	filterSize = np.asarray(weight.shape);
	filterHalfSize = [np.int(np.floor(filterSize[0]/2)), np.int(np.floor(filterSize[1]/2))]
	
	inputSize = np.asarray(X.shape);

	if (exponentsDict['input'] != 0):
		qX = scale(X, exponentsDict['input'])
		qW = scale(weight, exponentsDict['weight'])
		qB = scaleToInt32(bias, exponentsDict['input'] + exponentsDict['weight'])
		output = output.astype(np.int)

	for batch in range(outputSize[3]):
		for channel in range(outputSize[2]): #output channel
			for col in range(outputSize[1]):
				for row in range(outputSize[0]):
					

					if (exponentsDict['input'] !=0):
						out_pixel = qB[0][0][channel];
					else:
						out_pixel  = bias[0][0][channel];

					for r in range(filterSize[0]):
						for c in range(filterSize[1]):
							for ch in range(inputSize[2]): # input channels / features 
								
								# output row to input row index 
								# valid conv and offset by half filter size \
								# offset for filter mask position	
								input_row = (stride * row + 1) + \
									dilation * filterHalfSize[0] + \
									(dilation * (r - filterHalfSize[0])) - \
									padding[0]; # start frrorm top padding position
								input_row = input_row.astype(np.int)

								input_col = (stride * col + 1) + \
									dilation * filterHalfSize[1] + \
									(dilation * (c - filterHalfSize[1])) - \
									padding[2];
								input_col = input_col.astype(np.int)

								if input_row >= 0 and input_col >= 0 and input_row < inputSize[0] and input_col < inputSize[1]:
									if (exponentsDict['input'] !=0):
										a = qX[input_row][input_col][ch][batch]
										b = qW[r][c][ch][channel]
										out_pixel = out_pixel + np.int(a) * np.int(b);
									else:
										a =  X[input_row][input_col][ch][batch];
										b = W[r][c][ch][channel];
										out_pixel = out_pixel + a * b;
									
					output[row][col][channel][batch] = out_pixel;

	output = rescale(output, exponentsDict['input'] + exponentsDict['weight'])
	print(output.shape)
	return output;

if __name__ == "__main__":
	X = np.random.randint(-32, 31, (11, 11, 3, 1)); 
	W = np.random.normal(0, 1, (3, 3, 3, 64))
	b = np.ones((1, 1, 64))

	stride = 2
	padding = np.ones(4)
	dilation = 1
	
	exponentsDict = {'input': 0, 'weight': 0, 'bias': 0};

	floatConv = conv(X, W, b, stride, padding, dilation, exponentsDict)
	floatConv = floatConv.flatten();

	exponentsDict['input'] = computeExponent(X, 8);
	exponentsDict['weight'] = computeExponent(W, 8);

	print(exponentsDict)

	int8Conv = conv(X, W, b, stride, padding, dilation, exponentsDict)
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



