from numpy import *
import matplotlib.pyplot as plt
def loadFileData():
    dataArr=[];labelArr=[];
    fp=open('1_13points.txt')
    for line in fp.readlines():
        lineArr=line.strip().split(',')
        dataArr.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelArr.append(float(lineArr[2]))
        T,F=0,0
        for i in range(len(labelArr)):
            if labelArr[i]==1:
                T+=1
            else:
                F+=1
    return dataArr,labelArr, T,F

def sigmoid(y,m):
    # m for rescaling scale (F/T)
    return 1.0/(1+exp(-y)*m)


def plotFit(weights):
    n=shape(dataArr)[0]
    xcord1=[];ycord1=[]
    xcord2=[];ycord2=[]
    for i in range(n):
        if int(labelArr[i])==1:
            xcord1.append(dataArr[i][1])
            ycord1.append(dataArr[i][2])
        else:            
            xcord2.append(dataArr[i][1])
            ycord2.append(dataArr[i][2])
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker="s")
    ax.scatter(xcord2,ycord2,s=30,c='blue')
    x=arange(0,100.0,0.1)
    y=(-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x,y)
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()

def gradAscent(dataArr,labelArr):
    dataMat=mat(dataArr)
    labelMat=mat(labelArr).transpose()
    m,n=shape(dataMat)
    weights=ones((n,1))
    alpha=[]
    a=[1,1,1]    # add weight to different variables to train
    for i in range(n):
        x=zeros(n)
        for j in range(n):
            if i==j:
                x[i]=a[i]
        alpha.append(x)
    alpha=mat(alpha)
    accuracy=0
    global_accuracy=0
    total_error=0
    steps=0
    w0=[];w1=[];w2=[]
    # while accuracy<0.99:
    for i in range(50):
        if(global_accuracy<0.999):
            steps+=1                   # step for stabilization
        learning_rate = 0.1/(5.0+i)+0.01  # for decline of learning rate
        error=labelMat - sigmoid(dataMat*weights, 1.0*F/T)
        weights=weights + learning_rate*alpha*dataMat.transpose()*error   # w0 + w1*x + w2*y = 0
        w0.append(float(weights[0]))
        w1.append(float(weights[1]))
        w2.append(float(weights[2]))
        total_error=0      
        for i in range(n):
            total_error+=error[i]
        accuracy=1-abs(total_error)*1.0/n
        global_accuracy=0.7*global_accuracy+0.3*accuracy
        print("steps: %d, accuracy: %f"%(steps,global_accuracy))

    print(weights)       
    x=range(len(w0))
    plt.plot(x,w0)
    plt.plot(x,w1)
    plt.plot(x,w2)
    plt.title((a,steps))
    plt.show()
    return weights

def stocGradAscent(dataArr,labelArr):
    dataMat=mat(dataArr)
    m,n=shape(dataMat)
    weights=-1*ones(n)
    weights=mat(weights).transpose()
    print(weights)
    alpha=[]
    a=[10,1,1]    # add weight to different variables to train
    for i in range(n):
        x=zeros(n)
        for j in range(n):
            if i==j:
                x[i]=a[i]
        alpha.append(x)
    alpha=mat(alpha)
    accuracy=0
    global_accuracy=0
    total_error=0
    steps=0
    w0=[];w1=[];w2=[]
    for i in range(5):
        dataIndex=range(m) 
        for j in range(m):
            if(global_accuracy<0.999):
                steps+=1       # step for stabilization
            learning_rate = 2/(3.0+j+i)+0.01  # for decline of learning rate
            randIndex=int(random.uniform(0,len(dataIndex)))
            error=labelArr[randIndex] - sigmoid(float(dataArr[randIndex]*weights), 1.0*F/T)
            weights=weights + learning_rate*error*(alpha*dataMat[randIndex].transpose())   # w0 + w1*x + w2*y = 0
            del(dataIndex[randIndex])
            w0.append(weights[0,0])
            w1.append(weights[1,0])
            w2.append(weights[2,0])
            accuracy=1.0-abs(error)
            global_accuracy=0.8*global_accuracy+0.2*accuracy
            print("steps: %d, accuracy: %f"%(steps,global_accuracy))

    print(weights)       
    x=range(len(w0))
    plt.plot(x,w0)
    plt.plot(x,w1)
    plt.plot(x,w2)
    plt.title((a,steps))
    plt.show()
    return weights

dataArr,labelArr,T,F=loadFileData()
# weights=gradAscent(dataArr,labelArr)
weights=stocGradAscent(dataArr,labelArr)
plotFit(weights.getA())
