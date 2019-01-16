from numpy import *
import random
def random_points():
	for i in range(1000):
		a=random.uniform(1,99)
		b=random.uniform(-10,10)
		c=random.uniform(-5,5)
		d=random.uniform(-50,50)
		if a<c*b+d+2:
			label=0
		else:
			label=1
		print("%.1f,%.1f,%.1f,%.1f,%d"%(a,b,c,d,label))


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
