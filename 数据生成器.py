from numpy import *
import random
def random_points()
	for i in range(100):
		a=random.uniform(1,9)
		b=random.uniform(1,9)
		if a<2*b:
			c=1
		else:
			c=0
		print("%.1f,%.1f,%d"%(a,b+10,c))

def add_weight_for_train()
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

