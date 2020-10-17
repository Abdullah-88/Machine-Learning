import csv
import random
import math
import numpy
 
def dataprep(filename,numTrainingSet):
    lines = csv.reader(open(filename, "rb"),delimiter='\t')
    data = list(lines)
    temp = list(data)

    for i in range(len(temp)):
        line = temp[i]
        for j in range(1,len(line)-1 ):
            line[j] = float( line[j])
            temp[i]=line
    
    trainNumber = numTrainingSet
    trainSet = []
    validationSet = [] 
    for i in range(0,trainNumber):
        trainSet.append(temp[i])
    for i in range(trainNumber,len(temp)):
        validationSet.append(temp[i])
    return trainSet,validationSet



def classProbability(trainSet):
    classOneCount=0
    classTwoCount=0
    classThreeCount=0
    for i in range(len(trainSet)):
        s = trainSet[i]
        if s[8]=="1":
            classOneCount+=1
        elif s[8]=="2":
            classTwoCount+=1
        elif s[8]=="3":
            classThreeCount+=1

    c1Prob=classOneCount/float(len(trainSet))
    c2Prob=classTwoCount/float(len(trainSet))
    c3Prob=classThreeCount/float(len(trainSet))
    cProb=[c1Prob,c2Prob,c3Prob,classOneCount,classTwoCount,classThreeCount]
    return cProb



def genderProbability(classProb,trainSet):
    maleGivenOneCount=0
    femaleGivenOneCount=0
    iGivenOneCount=0
    maleGivenTwoCount=0
    femaleGivenTwoCount=0
    iGivenTwoCount=0
    maleGivenThreeCount=0
    femaleGivenThreeCount=0
    iGivenThreeCount=0
    for i in range(len(trainSet)):
        s = trainSet[i]
        if s[8]=="1":
            if s[0]=="M":
                maleGivenOneCount +=1
            elif s[0]=="F":
                femaleGivenOneCount +=1
            elif s[0]=="I":
                iGivenOneCount +=1
        elif s[8]=="2":
            if s[0]=="M":
                maleGivenTwoCount +=1
            elif s[0]=="F":
                femaleGivenTwoCount +=1
            elif s[0]=="I":
                iGivenTwoCount +=1
        elif s[8]=="3":
            if s[0]=="M":
                maleGivenThreeCount +=1
            elif s[0]=="F":
                femaleGivenThreeCount +=1
            elif s[0]=="I":
                iGivenThreeCount +=1
    maleGivenOneProb = maleGivenOneCount/float(classProb[3])
    femaleGivenOneProb = femaleGivenOneCount/float(classProb[3])
    iGivenOneProb = iGivenOneCount/float(classProb[3])
    maleGivenTwoProb = maleGivenTwoCount/float(classProb[4])
    femaleGivenTwoProb = femaleGivenTwoCount/float(classProb[4])
    iGivenTwoProb = iGivenTwoCount/float(classProb[4])
    maleGivenThreeProb = maleGivenThreeCount/float(classProb[5])
    femaleGivenThreeProb = femaleGivenThreeCount/float(classProb[5])
    iGivenThreeProb = iGivenThreeCount/float(classProb[5])
    genderProb=[[maleGivenOneProb,maleGivenTwoProb,maleGivenThreeProb],[femaleGivenOneProb,femaleGivenTwoProb,femaleGivenThreeProb],[iGivenOneProb,iGivenTwoProb,iGivenThreeProb]]
    return genderProb



def numerStats(trainSet):
    oneSubset=[]
    twoSubset=[]
    threeSubset=[]
    for i in range(len(trainSet)):
        L = trainSet[i]
        if L[8]=="1":
            oneSubset.append(L[1:len(L)-1])
        elif L[8]=="2":
            twoSubset.append(L[1:len(L)-1])
        elif L[8]=="3":
            threeSubset.append(L[1:len(L)-1])

        
    a1 = numpy.array(oneSubset)
    meanGivenOne=numpy.mean(a1,axis=0).tolist()
    stdvGivenOne=numpy.std(a1,axis=0).tolist()
    a2=numpy.array(twoSubset)
    meanGivenTwo=numpy.mean(a2,axis=0).tolist()
    stdvGivenTwo=numpy.std(a2,axis=0).tolist()
    a3=numpy.array(threeSubset)
    meanGivenThree=numpy.mean(a3,axis=0).tolist()
    stdvGivenThree=numpy.std(a3,axis=0).tolist()
    nStats=[meanGivenOne,stdvGivenOne,meanGivenTwo,stdvGivenTwo,meanGivenThree,stdvGivenThree]
    return nStats



def calculateProbability(x, mean, stdev):
	exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
	return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent    
     
    

def numerAttributeProbabilityFull(inSet,stats):
    gProb=[]
    for i in range(0,len(inSet)):
        L=inSet[i]
        gProbTemp=[]
        for j in range(1,len(L)-1):
           gProbTemp.append([calculateProbability(L[j],stats[0][j-1],stats[1][j-1]),calculateProbability(L[j],stats[2][j-1],stats[3][j-1]),calculateProbability(L[j],stats[4][j-1],stats[5][j-1])])
        gProb.append(gProbTemp)
    numerProb=[]
    for i in range(0,len(gProb)):
        g=gProb[i]
        t1=1
        t2=1
        t3=1
        for j in range(0,len(g)):
            t1*=g[j][0]
            t2*=g[j][1]
            t3*=g[j][2]
        numerProb.append([t1,t2,t3])
    return  numerProb



     
    

def numerAttributeProbabilityNonFull(inSet,stats):
    gProb=[]
    for i in range(0,len(inSet)):
        L=inSet[i]
        gProbTemp=[]
        for j in range(1,len(L)-1):
           gProbTemp.append([calculateProbability(L[j],stats[0][j-1],stats[1][j-1]),calculateProbability(L[j],stats[2][j-1],stats[3][j-1]),calculateProbability(L[j],stats[4][j-1],stats[5][j-1])])
        gProb.append(gProbTemp)
    numerProb=[]
    for i in range(0,len(gProb)):
        g=gProb[i]
        t1=1
        t2=1
        t3=1
        for j in range(0,2):
            t1*=g[j][0]
            t2*=g[j][1]
            t3*=g[j][2]
        numerProb.append([t1,t2,t3])
    return  numerProb




def prediction(classProb,genderProb,inSet,numerProb):
    pred=[]
    for i in range(0,len(inSet)):
        if inSet[i][0]=="M":
            lh1=classProb[0]*genderProb[0][0]*numerProb[i][0]
            lh2=classProb[1]*genderProb[0][1]*numerProb[i][1]
            lh3=classProb[2]*genderProb[0][2]*numerProb[i][2]
            if lh1==max(lh1,lh2,lh3):
                pred.append("1")
            elif lh2==max(lh1,lh2,lh3):
                pred.append("2")
            elif lh3==max(lh1,lh2,lh3):
                pred.append("3")    
        elif inSet[i][0]=="F":
            lh1=classProb[0]*genderProb[1][0]*numerProb[i][0]
            lh2=classProb[1]*genderProb[1][1]*numerProb[i][1]
            lh3=classProb[2]*genderProb[1][2]*numerProb[i][2]
            if lh1==max(lh1,lh2,lh3):
                pred.append("1")
            elif lh2==max(lh1,lh2,lh3):
                pred.append("2")    
            elif lh3==max(lh1,lh2,lh3):
                pred.append("3")     
        elif inSet[i][0]=="I":
            lh1=classProb[0]*genderProb[2][0]*numerProb[i][0]
            lh2=classProb[1]*genderProb[2][1]*numerProb[i][1]
            lh3=classProb[2]*genderProb[2][2]*numerProb[i][2]
            if lh1==max(lh1,lh2,lh3):
                pred.append("1")
            elif lh2==max(lh1,lh2,lh3):
                pred.append("2")    
            elif lh3==max(lh1,lh2,lh3):
                pred.append("3")
    return pred            



def misclassificationError(inSet,pred):
    misses=0
    for i in range(0,len(inSet)):
        if pred[i]!= inSet[i][8]:
            misses+=1
    return misses


def accuracy(inSet,pred):
    exact = 0
    for i in range(len(inSet)):
        if pred[i]==inSet[i][8]:
            exact+= 1
    accur=(exact/float(len(inSet))) * 100.0
    return  accur


def confusionMatrix(pred,inSet):
    oneone=0
    onetwo=0
    onethree=0
    twoone=0
    twotwo=0
    twothree=0
    threeone=0
    threetwo=0
    threethree=0
    
    for i in range(0,len(inSet)):
        if pred[i]=="1":
            if inSet[i][8]=="1":
                oneone+=1
            elif inSet[i][8]=="2":
                onetwo+=1
            elif inSet[i][8]=="3":
                onethree+=1
        elif pred[i]=="2":
            if inSet[i][8]=="1":
                twoone+=1
            elif inSet[i][8]=="2":
                twotwo+=1
            elif inSet[i][8]=="3":
                twothree+=1
        elif pred[i]=="3":
            if inSet[i][8]=="1":
                threeone+=1
            elif inSet[i][8]=="2":
                threetwo+=1
            elif inSet[i][8]=="3":
                threethree+=1
    contMat=[[oneone,onetwo,onethree],[twoone,twotwo,twothree],[threeone,threetwo,threethree]]
    contMat=numpy.array(contMat)
    return contMat
            
        
          
def testingNonFull():
    print "\n Using First Three Attributes \n"
    # 100 training samples
    print "\n Using 100 training samples \n"
    trainSet,validationSet=dataprep("\\Users\\dc\\Desktop\\abalone_dataset.txt",100)
    classProb= classProbability(trainSet)
    genderProb=genderProbability(classProb,trainSet)
    stats = numerStats(trainSet)
    numerProbTrain=numerAttributeProbabilityNonFull(trainSet,stats)
    numerProbValidation=numerAttributeProbabilityNonFull(validationSet,stats)
    predTrain=prediction(classProb,genderProb,trainSet,numerProbTrain)
    predValidation=prediction(classProb,genderProb,validationSet,numerProbValidation)
    #print "\n Predictions for training set: \n"
    #print predTrain
    print "\n miss classification error for training set \n"
    print misclassificationError(trainSet,predTrain)
    print "\n Accuracy for training set \n"
    print accuracy(trainSet,predTrain)
    print "\n Confusion Matrix for training set \n"
    print confusionMatrix(predTrain,trainSet)
    #print "\n Predictions for validation set: \n"
    #print predValidation
    print "\n miss classification error for Validation set \n"
    print misclassificationError(validationSet,predValidation)
    print "\n Accuracy for validation set \n"
    print accuracy(validationSet,predValidation)
    print "\n Confusion Matrix for validation set \n"
    print confusionMatrix(predValidation,validationSet)
    # 1000 training samples
    print "\n Using 1000 training samples \n"
    trainSet,validationSet=dataprep("\\Users\\dc\\Desktop\\abalone_dataset.txt",1000)
    classProb= classProbability(trainSet)
    genderProb=genderProbability(classProb,trainSet)
    stats = numerStats(trainSet)
    numerProbTrain=numerAttributeProbabilityNonFull(trainSet,stats)
    numerProbValidation=numerAttributeProbabilityNonFull(validationSet,stats)
    predTrain=prediction(classProb,genderProb,trainSet,numerProbTrain)
    predValidation=prediction(classProb,genderProb,validationSet,numerProbValidation)
    #print "\n Predictions for training set: \n"
    #print predTrain
    print "\n miss classification error for training set \n"
    print misclassificationError(trainSet,predTrain)
    print "\n Accuracy for training set \n"
    print accuracy(trainSet,predTrain)
    print "\n Confusion Matrix for training set \n"
    print confusionMatrix(predTrain,trainSet)
    #print "\n Predictions for validation set: \n"
    #print predValidation
    print "\n miss classification error for Validation set \n"
    print misclassificationError(validationSet,predValidation)
    print "\n Accuracy for validation set \n"
    print accuracy(validationSet,predValidation)
    print "\n Confusion Matrix for validation set \n"
    print confusionMatrix(predValidation,validationSet)
    # 2000 training samples
    print "\n Using 2000 training samples \n"
    trainSet,validationSet=dataprep("\\Users\\dc\\Desktop\\abalone_dataset.txt",2000)
    classProb= classProbability(trainSet)
    genderProb=genderProbability(classProb,trainSet)
    stats = numerStats(trainSet)
    numerProbTrain=numerAttributeProbabilityNonFull(trainSet,stats)
    numerProbValidation=numerAttributeProbabilityNonFull(validationSet,stats)
    predTrain=prediction(classProb,genderProb,trainSet,numerProbTrain)
    predValidation=prediction(classProb,genderProb,validationSet,numerProbValidation)
    #print "\n Predictions for training set: \n"
    #print predTrain
    print "\n miss classification error for training set \n"
    print misclassificationError(trainSet,predTrain)
    print "\n Accuracy for training set \n"
    print accuracy(trainSet,predTrain)
    print "\n Confusion Matrix for training set \n"
    print confusionMatrix(predTrain,trainSet)
    #print "\n Predictions for validation set: \n"
    #print predValidation
    print "\n miss classification error for Validation set \n"
    print misclassificationError(validationSet,predValidation)
    print "\n Accuracy for validation set \n"
    print accuracy(validationSet,predValidation)
    print "\n Confusion Matrix for validation set \n"
    print confusionMatrix(predValidation,validationSet)

def testingFull():
    print "\n Using Full Number Of Attributes \n"
    # 100 training samples
    print "\n Using 100 training samples \n"
    trainSet,validationSet=dataprep("\\Users\\dc\\Desktop\\abalone_dataset.txt",100)
    classProb= classProbability(trainSet)
    genderProb=genderProbability(classProb,trainSet)
    stats = numerStats(trainSet)
    numerProbTrain=numerAttributeProbabilityFull(trainSet,stats)
    numerProbValidation=numerAttributeProbabilityFull(validationSet,stats)
    predTrain=prediction(classProb,genderProb,trainSet,numerProbTrain)
    predValidation=prediction(classProb,genderProb,validationSet,numerProbValidation)
    #print "\n Predictions for training set: \n"
    #print predTrain
    print "\n miss classification error for training set \n"
    print misclassificationError(trainSet,predTrain)
    print "\n Accuracy for training set \n"
    print accuracy(trainSet,predTrain)
    print "\n Confusion Matrix for training set \n"
    print confusionMatrix(predTrain,trainSet)
    #print "\n Predictions for validation set: \n"
    #print predValidation
    print "\n miss classification error for Validation set \n"
    print misclassificationError(validationSet,predValidation)
    print "\n Accuracy for validation set \n"
    print accuracy(validationSet,predValidation)
    print "\n Confusion Matrix for validation set \n"
    print confusionMatrix(predValidation,validationSet)
    # 1000 training samples
    print "\n Using 1000 training samples \n"
    trainSet,validationSet=dataprep("\\Users\\dc\\Desktop\\abalone_dataset.txt",1000)
    classProb= classProbability(trainSet)
    genderProb=genderProbability(classProb,trainSet)
    stats = numerStats(trainSet)
    numerProbTrain=numerAttributeProbabilityFull(trainSet,stats)
    numerProbValidation=numerAttributeProbabilityFull(validationSet,stats)
    predTrain=prediction(classProb,genderProb,trainSet,numerProbTrain)
    predValidation=prediction(classProb,genderProb,validationSet,numerProbValidation)
    #print "\n Predictions for training set: \n"
    #print predTrain
    print "\n miss classification error for training set \n"
    print misclassificationError(trainSet,predTrain)
    print "\n Accuracy for training set \n"
    print accuracy(trainSet,predTrain)
    print "\n Confusion Matrix for training set \n"
    print confusionMatrix(predTrain,trainSet)
    #print "\n Predictions for validation set: \n"
    #print predValidation
    print "\n miss classification error for Validation set \n"
    print misclassificationError(validationSet,predValidation)
    print "\n Accuracy for validation set \n"
    print accuracy(validationSet,predValidation)
    print "\n Confusion Matrix for validation set \n"
    print confusionMatrix(predValidation,validationSet)
    # 2000 training samples
    print "\n Using 2000 training samples \n"
    trainSet,validationSet=dataprep("\\Users\\dc\\Desktop\\abalone_dataset.txt",2000)
    classProb= classProbability(trainSet)
    genderProb=genderProbability(classProb,trainSet)
    stats = numerStats(trainSet)
    numerProbTrain=numerAttributeProbabilityFull(trainSet,stats)
    numerProbValidation=numerAttributeProbabilityFull(validationSet,stats)
    predTrain=prediction(classProb,genderProb,trainSet,numerProbTrain)
    predValidation=prediction(classProb,genderProb,validationSet,numerProbValidation)
    #print "\n Predictions for training set: \n"
    #print predTrain
    print "\n miss classification error for training set \n"
    print misclassificationError(trainSet,predTrain)
    print "\n Accuracy for training set \n"
    print accuracy(trainSet,predTrain)
    print "\n Confusion Matrix for training set \n"
    print confusionMatrix(predTrain,trainSet)
    #print "\n Predictions for validation set: \n"
    #print predValidation
    print "\n miss classification error for Validation set \n"
    print misclassificationError(validationSet,predValidation)
    print "\n Accuracy for validation set \n"
    print accuracy(validationSet,predValidation)
    print "\n Confusion Matrix for validation set \n"
    print confusionMatrix(predValidation,validationSet)    
    

testingNonFull()
testingFull()



