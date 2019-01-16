from numpy import *
import random
def random_points():
	for i in range(1000):
		a=random.uniform(1,99)
		b=random.uniform(1,99)
		if a<5*b+2:
			c=1
		else:
			c=0
		print("%.1f,%.1f,%d"%(a,b,c))

def add_weight_for_train():
	m=5
	a=[1,2,3,4,5]
	alpha=[]
	for i in range(m):
		x=zeros(m)
		for j in range(m):
			if i==j:
				print(i,j,a[i])
				x[i]=a[i]
		alpha.append(x)
	alpha=mat(alpha)
	print(alpha)

random_points()
