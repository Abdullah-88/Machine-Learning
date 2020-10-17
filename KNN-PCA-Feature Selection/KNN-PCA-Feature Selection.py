import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import itertools


def accuracyScore(actual,predicted):
    sum=0
    for i in range(len(actual)):
        if actual[i]==predicted[i]:
            sum+=1
    return (sum/float(len(actual)))*100

def precisionScore(actual,predicted):
    sum1=0
    sum2=0
    for i in range(len(actual)):
       if actual[i]==1 and predicted[i]==1:
           sum1+=1
       if actual[i]==0 and predicted[i]==1:
           sum2+=1
    return (sum1 / float(sum1+sum2))*100

def recallScore(actual,predicted):
    sum1 = 0
    sum2 = 0
    for i in range(len(actual)):
        if actual[i] == 1 and predicted[i] == 1:
            sum1 += 1
        if actual[i] == 1 and predicted[i] == 0:
            sum2 += 1
    return (sum1 / float(sum1+sum2))*100


def dataPrep(fileName):
    splitRatio = 0.5
    dt = pd.read_csv(fileName)
    df = pd.DataFrame(dt)
    df=df.reindex(np.random.permutation(df.index))
    df.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
    trainDF = df.iloc[0:int((len(df)*splitRatio))]
    validationDF =  df.iloc[int((len(df)*splitRatio)):int(len(df))]

    trainspamLabel=np.array(trainDF.iloc[:,-1]).tolist()

    validationspamLabel= np.array(validationDF.iloc[:,-1]).tolist()

    trainAttributes=np.array(trainDF.iloc[:,0:-1]).tolist()
    validationAttributes=np.array(validationDF.iloc[:,0:-1]).tolist()

    return trainspamLabel,trainAttributes,validationspamLabel,validationAttributes

trainspamLabel,trainAttributes,validationspamLabel,validationAttributes=dataPrep('dataset.txt')

def KNN(inputAttributes,modelAttributes,modelLabel,k):
    predictions=[]
    for m in range(len(inputAttributes)):
        iAttr=inputAttributes[m]
        distances = []
        NNLabels=[]
        for i in range(len(modelAttributes)):
            att=modelAttributes[i]
            sum=0
            for j in range(len(att)):
                sum+=np.square(iAttr[j]-att[j])
            distance=np.sqrt(sum)
            distances.append([distance,i])

        distances=sorted(distances)

        for n in range(k):

            NNLabels.append(modelLabel[distances[n][1]])

        counts = Counter(NNLabels)
        pred=counts.most_common(1)

        predictions.append(pred[0][0])
    return predictions


def PCA(inputAttributes,M):
    covMat=np.cov(np.array(inputAttributes).T)
    U, s, V = np.linalg.svd(covMat, full_matrices=True)
    total = np.sum(s)
    variance = np.array([np.sum(s[: i + 1]) / total * 100.0 for i in range(len(trainAttributes[0]))])
    """plt.plot(variance)
    plt.xlabel('Number of PCA components')
    plt.ylabel(' Percentage Variance retained')
    plt.title('Variance vs PCA components')
    plt.show()"""
    projMat=U[:,:M]
    proj = np.array(inputAttributes).dot(projMat)
    return proj

def cForwardSelection(inputAttributes,modelAttributes,modelLabels,validationLabels,K,M):
    accuracyList=[]
    combinations = np.array(list(itertools.combinations(range(len(modelAttributes[0])), M)))
    print (len(combinations))
    for i in range(len(combinations)):
        combtemp=combinations[i]
        itempDF=pd.DataFrame()
        mtempDF=pd.DataFrame()

        for j in range(len(combtemp)):
            itempDF=itempDF.append(pd.DataFrame(inputAttributes).iloc[:, combtemp[j]:combtemp[j]+1])
            mtempDF=mtempDF.append(pd.DataFrame(modelAttributes).iloc[:, combtemp[j]:combtemp[j]+1])

        inputselectedFeature = np.array(itempDF).tolist()

        modelselectedFeature = np.array(mtempDF).tolist()
        print(pd.DataFrame(inputselectedFeature).shape)
        pred = KNN(inputselectedFeature, modelselectedFeature, modelLabels, K)
        print(pred)
        accuracy = accuracyScore(validationLabels, pred)
        print(accuracy)
        accuracyList.append([accuracy, inputselectedFeature,modelselectedFeature])
    accuracyList.sort()
    accuracyList.reverse()
    validationsetNew=accuracyList[0][1]
    trainsetNew=accuracyList[0][2]
    return trainsetNew, validationsetNew


def sForwardSelection(inputAttributes,modelAttributes,modelLabels,validationLabels,K,M):
    accuracyList=[]
    itempDF=pd.DataFrame()
    mtempDF=pd.DataFrame()
    for m in range(M):

        for j in range(len(pd.DataFrame(modelAttributes[0]))):
            inputfeature = np.array(itempDF.append(pd.DataFrame(inputAttributes).iloc[:, j:j+1])).tolist()
            modelfeature = np.array(mtempDF.append(pd.DataFrame(modelAttributes).iloc[:, j:j+1])).tolist()
            print(pd.DataFrame(inputfeature).shape)
            pred = KNN(inputfeature, modelfeature, modelLabels, K)
            accuracy = accuracyScore(validationLabels, pred)
            print(accuracy)
            accuracyList.append([accuracy, j])
        accuracyList.sort()
        accuracyList.reverse()
        print(accuracyList)
        itempDF=itempDF.append(pd.DataFrame(inputAttributes).iloc[:, accuracyList[0][1]:accuracyList[0][1]+1])
        inputAttributes=np.array(pd.DataFrame(inputAttributes).drop([ accuracyList[0][1]],1))
        mtempDF=mtempDF.append(pd.DataFrame(modelAttributes).iloc[:, accuracyList[0][1]:accuracyList[0][1]:accuracyList[0][1]+1])
        modelAttributes=np.array(pd.DataFrame(modelAttributes).drop([ accuracyList[0][1]],1))
    return np.array(mtempDF).tolist(),np.array(itempDF).tolist()





def main():

    trainspamLabel,trainAttributes,validationspamLabel,validationAttributes=dataPrep('dataset.txt')
    k=5

    predictions = KNN(trainAttributes, trainAttributes, trainspamLabel, k)
    accuracy = accuracyScore(trainspamLabel, predictions)
    precision = precisionScore(trainspamLabel, predictions)
    recall = recallScore(trainspamLabel, predictions)
    print("KNN without feature selection and extraction\n")
    print("\n Train results")
    print("\n Accuracy : ")
    print(accuracy)
    print("\n Precision : ")
    print(precision)
    print("\n Recall : ")
    print(recall)

    predictions=KNN(validationAttributes,trainAttributes,trainspamLabel,k)
    accuracy=accuracyScore(validationspamLabel,predictions)
    precision=precisionScore(validationspamLabel,predictions)
    recall=recallScore(validationspamLabel,predictions)
    print("KNN without feature selection and extraction\n")
    print("\n Validation results")
    print("\n Accuracy : ")
    print(accuracy)
    print("\n Precision : ")
    print(precision)
    print("\n Recall : ")
    print(recall)

    projected=PCA(validationAttributes,2)
    modelprojected=PCA(trainAttributes,2)
    print("\n DATA on PCA m=2")
    print("\n Validation")
    print(pd.DataFrame(projected))
    print("\n Train")
    print(pd.DataFrame(modelprojected))
    pred=KNN(projected,modelprojected,trainspamLabel,5)
    accuracy=accuracyScore(validationspamLabel,pred)
    precision = precisionScore(validationspamLabel, pred)
    recall = recallScore(validationspamLabel, pred)
    



    print("\n PCA+KNN Train")
    accBin=[]
    presBin=[]
    recBin=[]
    for m in range(1,58):
        validationprojected = PCA(validationAttributes, m)
        modelprojected = PCA(trainAttributes, m)
        pred = KNN(modelprojected, modelprojected, trainspamLabel, 5)
        print('check')
        accuracy = accuracyScore(trainspamLabel, pred)
        precision = precisionScore(trainspamLabel, pred)
        recall = recallScore(trainspamLabel, pred)
        accBin.append(accuracy)
        presBin.append(precision)
        recBin.append(recall)
    
    plt.plot(accBin)
    plt.xlabel('Number of PCA components')
    plt.ylabel('accuracy')
    plt.title('Train Accuracy vs PCA components')
    plt.show()

    plt.plot(presBin)
    plt.xlabel('Number of PCA components')
    plt.ylabel('precission')
    plt.title('Train Precission vs PCA components')
    plt.show()
    plt.plot(recBin)
    plt.xlabel('Number of PCA components')
    plt.ylabel('recall')
    plt.title('Train Recall vs PCA components')
    plt.show()
    accBin.sort()
    accBin.reverse()
    print('\n Highest Train Accuracy on PCA+KNN Train:')
    print(accBin[0])
    
    print("\n PCA+KNN Validation")
    accBin = []
    presBin = []
    recBin = []
    for m in range(1,58):
        validationprojected = PCA(validationAttributes, m)
        modelprojected = PCA(trainAttributes, m)
        pred = KNN(validationprojected, modelprojected, trainspamLabel, 5)
        print('check')
        accuracy = accuracyScore(validationspamLabel, pred)
        precision = precisionScore(validationspamLabel, pred)
        recall = recallScore(validationspamLabel, pred)
        accBin.append(accuracy)
        presBin.append(precision)
        recBin.append(recall)
   
    plt.plot(accBin)
    plt.xlabel('Number of PCA components')
    plt.ylabel('accuracy')
    plt.title('Validation Accuracy vs PCA components')
    plt.show()

    plt.plot(presBin)
    plt.xlabel('Number of PCA components')
    plt.ylabel('precission')
    plt.title('Validation Precission vs PCA components')
    plt.show()
    plt.plot(recBin)
    plt.xlabel('Number of PCA components')
    plt.ylabel('recall')
    plt.title('Validation Recall vs PCA components')
    plt.show()
    accBin.sort()
    accBin.reverse()
    print('\n Highest Validation Accuracy on PCA+KNN validation:')
    print(accBin[0])
    newTrain,newValidation=sForwardSelection(validationAttributes,trainAttributes,trainspamLabel,validationspamLabel,5,2)
    print(pd.DataFrame(newTrain))
    print(pd.DataFrame(newValidation))
    print("\n PCA+FS Train")
    accBin = []
    presBin = []
    recBin = []
    for m in range(1, 58):
        modelselected,validationselected=cForwardSelection(validationAttributes,trainAttributes,trainspamLabel,validationspamLabel,5,m)
        pred = KNN(modelselected, modelselected, trainspamLabel, 5)
        print('check')
        accuracy = accuracyScore(trainspamLabel, pred)
        precision = precisionScore(trainspamLabel, pred)
        recall = recallScore(trainspamLabel, pred)
        accBin.append(accuracy)
        presBin.append(precision)
        recBin.append(recall)

    plt.plot(accBin)
    plt.xlabel('Number of selected features')
    plt.ylabel('accuracy')
    plt.title('Train Accuracy vs selected features')
    plt.show()

    plt.plot(presBin)
    plt.xlabel('Number of selected features')
    plt.ylabel('precission')
    plt.title('Train Precission vs selected features')
    plt.show()
    plt.plot(recBin)
    plt.xlabel('Number of selected features')
    plt.ylabel('recall')
    plt.title('Train Recall vs selected features')
    plt.show()
    accBin.sort()
    accBin.reverse()
    print('\n Highest Validation Accuracy on PCA+FS training:')
    print(accBin[0])

    print("\n PCA+FS Validation")
    accBin = []
    presBin = []
    recBin = []
    for m in range(1, 58):
        modelselected,validationselected=cForwardSelection(validationAttributes,trainAttributes,trainspamLabel,validationspamLabel,5,m)
        pred = KNN(validationselected, modelselected, trainspamLabel, 5)
        print('check')
        accuracy = accuracyScore(trainspamLabel, pred)
        precision = precisionScore(trainspamLabel, pred)
        recall = recallScore(trainspamLabel, pred)
        accBin.append(accuracy)
        presBin.append(precision)
        recBin.append(recall)

    plt.plot(accBin)
    plt.xlabel('Number of selected features')
    plt.ylabel('accuracy')
    plt.title('validation Accuracy vs selected features')
    plt.show()

    plt.plot(presBin)
    plt.xlabel('Number of selected features')
    plt.ylabel('precission')
    plt.title('validation Precission vs selected features')
    plt.show()
    plt.plot(recBin)
    plt.xlabel('Number of selected features')
    plt.ylabel('recall')
    plt.title('validation Recall vs selected features')
    plt.show()
    accBin.sort()
    accBin.reverse()
    print('\n Highest Validation Accuracy on PCA+FS validation:')
    print(accBin[0])

main()