from numpy import *
import matplotlib.pyplot as plt
def loadFileData():
    dataArrTrain=[];labelArrTrain=[];
    dataArrTest=[];labelArrTest=[];
    fp=open('1_13points.txt')
    T,F=0,0
    for line in fp.readlines():
        lineArr=line.strip().split(',')
        lineArr = [ float(x) for x in lineArr ]
        temp=lineArr[0:-1]
        temp.insert(0,1.0)
        if int(lineArr[-1])==1:
            T+=1
        else:
            F+=1
        if (T+F)%5:
            dataArrTrain.append(temp)
            labelArrTrain.append(int(lineArr[-1]))
        else:
            dataArrTest.append(temp)
            labelArrTest.append(int(lineArr[-1]))
                
    return dataArrTrain,labelArrTrain,dataArrTest,labelArrTest, T,F

def sigmoid(y,m):
    # m for rescaling scale (F/T)
    return 1.0/(1+exp(-y)*m)


def plotFit(weights, dataArr, labelArr):
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

def gradAscent(dataArrTrain,labelArrTrain):
    dataMat=mat(dataArrTrain)
    labelMat=mat(labelArrTrain).transpose()
    m,n=shape(dataMat)
    weights=ones((n,1))
    alpha=[]
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
    w=[]
    for i in range(n):
        w.append([])
    # while accuracy<0.99:
    for i in range(50):
        if(global_accuracy<0.999):
            steps+=1                   # step for stabilization
        learning_rate = 0.1/(5.0+i)+0.01  # for decline of learning rate
        error=labelMat - sigmoid(dataMat*weights, 1.0*F/T)
        weights=weights + learning_rate*alpha*dataMat.transpose()*error   # w0 + w1*x + w2*y = 0
        total_error=0
        for i in range(n):
            w[i].append(float(weights[i]))
            total_error+=error[i]
        accuracy=1-abs(total_error)*1.0/n
        global_accuracy=0.7*global_accuracy+0.3*accuracy
        print("steps: %d, accuracy: %f"%(steps,global_accuracy))

    print(weights)       
    x=range(len(w[0]))
    for i in range(n):
        plt.plot(x,w[i])
    plt.title((a,steps))
    plt.show()
    return weights

def stocGradAscent(dataArrTrain,labelArrTrain):
    dataMat=mat(dataArrTrain)
    m,n=shape(dataMat)
    weights=-1*ones(n)
    weights=mat(weights).transpose()
    alpha=[]
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
    w=[]
    lr=[]
    for i in range(n):
        w.append([])
    for i in range(50):
        dataIndex=range(m) 
        learning_rate = 0.1/(5.0+i)+0.001  # for decline of learning rate
        for j in range(m):
            if(global_accuracy<0.999):
                steps+=1       # step for stabilization
            randIndex=int(random.uniform(0,len(dataIndex)))
            error=labelArrTrain[randIndex] - sigmoid(float(dataArrTrain[randIndex]*weights), 1.0*F/T)
            weights=weights + learning_rate*error*(alpha*dataMat[randIndex].transpose())   # w0 + w1*x + w2*y = 0
            del(dataIndex[randIndex])
            for i in range(n):
                w[i].append(float(weights[i]))
            lr.append(learning_rate)
            accuracy=1.0-abs(error)
            global_accuracy=0.8*global_accuracy+0.2*accuracy
            # print("steps: %d, accuracy: %f"%(steps,global_accuracy))

    print(weights)       
    x=range(len(w[0]))
    for i in range(n):
        plt.plot(x,w[i])
    plt.title((a,steps))
    plt.show()    
    plt.plot(x,lr)
    plt.show()
    return weights

def test(dataArrTest,labelArrTest):
    dataMat=mat(dataArrTest)
    labelMat=mat(labelArrTest).transpose()
    error=labelMat - sigmoid(dataMat*weights, 1.0)
    accuracy=1-abs(sum(error))/len(labelArrTest)
    print("accuracy: %f"%accuracy)
    plotFit(weights.getA(), dataArrTest,labelArrTest)

a=[1,1,1,1,1]    # add weight to different variables to train
dataArrTrain,labelArrTrain,dataArrTest,labelArrTest,T,F=loadFileData()
weights=gradAscent(dataArrTrain,labelArrTrain)
# weights=stocGradAscent(dataArrTrain,labelArrTrain)
plotFit(weights.getA(),dataArrTrain,labelArrTrain)
test(dataArrTest,labelArrTest)
print(T,F)
