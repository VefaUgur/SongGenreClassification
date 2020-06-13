import numpy as np
import operator
import pandas as pd

def calculateRecommendSuccess(dataPath,labelPath):
    def load_data(data,genre):
        a = np.load(data, allow_pickle = True)
        b = np.load(genre,allow_pickle = True)
        return a, b
    def euclideanDistance(instance1, instance2, length):
    	distance = 0
    	for x in range(length):
    		distance += pow((instance1[x] - instance2[x]), 2)
    	return np.sqrt(distance)
    
    def find_knn(k,id,features):
        dist=[]
        for i in range(len(features)):
            d=euclideanDistance(features[id],features[i],len(features[0]))
            dist.append((i,d))
        dist.sort(key=operator.itemgetter(1))
        return dist[1:k+1]
            
    def recommend(data,names, ind,k=5):
        acc = 0
        r=find_knn(k,ind,data)
        for i in r:
            if(names[i[0]]==names[ind]):
                acc+=1
        return acc / k
    def total_acc(data,names,k):
        table = []
        acc = 0
        t_acc = 0 
        for i in range(len(data)):
            tmp =recommend(data,names,i,k)
            acc += tmp
            t_acc += tmp
            if(i%100==99):
                print(k,'->',names[i],': ',acc/100)
                table.append([names[i],acc/100])
                acc = 0
        print('total acc: ',t_acc/1000)
        table.append(['total_acc',t_acc/1000])
        return table
    
    #data, names = load_data('FeatureExtractData.npy','FeatureExtractLabel.npy')
    data, names = load_data(dataPath,labelPath)
    
    k = [5,10,15]
    table = []
    for i in k:
        table.append(total_acc(data,names,i))
    np.save('table_39D',np.array(table))