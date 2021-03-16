import numpy as np
import matplotlib.pyplot as plt

def random_dice(N):
	a = np.random.rand(N)
	b = np.floor((6*a + 1))
	return b

def reshape_vector(y): 
	arr = np.array(y)
	a = arr.reshape(3,2)
	return a

def count_ones(v):
	a = np.array(v)
	elmnt, ct = np.unique(a, return_counts=True)
	b = dict(zip(elmnt, ct))
	x = b[1]
	return x

def max_value(z):
	arr = np.array(z)
	x = np.amax(arr)
	first_index = np.argwhere(arr==x)[0]
	r = first_index[0]
	c = first_index[1]
	return (x,r,c)

if __name__ == '__main__':
	dice = random_dice(10)
	reshaped = reshape_vector([11,22,33,44,55,66])
	max_value = max_value([[1,2],[2,5],[8,1]])
	countones = count_ones([[1,2],[2,5],[8,1]])
	print(dice)
	print(reshaped)
	print(max_value)
	print(countones)



