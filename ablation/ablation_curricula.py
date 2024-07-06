import numpy as np 
from fibonacci import fibonacci
import math

def static(max_mask, num_epochs):
	return [max_mask] * num_epochs

def linear(min_mask, max_mask, num_epochs):
	v = np.linspace(min_mask, max_mask, num_epochs//4)
	temp=[]
	for i in v:
		for j in range(4):
			temp.append(i)
	return temp

def log(max_mask, num_epochs):
	v = []
	for i in range(1,num_epochs//4+1):
		v.append(math.log(i)/(math.log(num_epochs//4)/max_mask))
	temp=[]
	for i in v:
		for j in range(4):
			temp.append(i)
	return temp

def exp(max_mask, num_epochs, regularization = 0.148):
	ex3=[]
	E=2.718281828459045
	for i in range(num_epochs//4):
		ex3.append((E**(regularization*i))/(E**(regularization*(num_epochs/4-1))/max_mask))
	temp=[]
	for i in ex3:
		for j in range(4):
			temp.append(i)
	return temp

def anti_linear(min_mask, max_mask, num_epochs):
	v = np.linspace(min_mask, max_mask, num_epochs//4)
	temp=[]
	for i in v:
		for j in range(4):
			temp.append(i)
	return temp[::-1]

def linear_repeat(min_mask, max_mask, num_epochs):
	v = fibonacci(length=7)
	for i in range(1,len(v)):
		v[i] = math.log(v[i])/(math.log(v[6])/max_mask)
	v = v[2:]
	v[0] = min_mask
	return v * (num_epochs//5)

def baseline(num_epochs):
	return [0] * num_epochs
