# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 13:38:19 2020

@author: VEFAUĞURBİLGE
"""


import tensorflow as tf
import pandas as pd 
import numpy as np
import gc
from scipy.spatial.distance import pdist, squareform

def recommend(dataType):
    #Load CNN-2 Model without last Layer
    model = tf.keras.models.load_model("spectrograms_CNN2_feature_checkpoint.h5",compile=False)
    intermediate_layer_model = tf.keras.Model(inputs=model.input,outputs=model.get_layer('dropout_2').output)
    intermediate_layer_model.summary()
    
    def getDataWithLabel():
        data = pd.read_pickle("DataSets/"+dataType+"_data.pkl") 
        #data = data.sample(frac=1).reset_index(drop=True)
        # put labels into y_train variable
        labels = data['Category']
        # Drop 'label' column
        data = data.drop(labels = ['Category'],axis = 1) 
        return data,labels
    
    def createTestAndTrain():
         data,label = getDataWithLabel()
         # Normalize the data
         data= data.astype('float16')
         data = data / 255.0
         print("Data was normalized..")
         print("Data shape: ",data.shape)
         #Reshape to matrix
         data = data.values.reshape(-1,240,240,3)
         print("Data was reshaped..")
        # label = to_categorical(label, num_classes = 10)
         return data, label ;
     
    def featureExtract():
         #Get Dataset
        data,label= createTestAndTrain()
        data.shape
        
        #Feature Extraction create new dataset
        feauture_engg_data = intermediate_layer_model.predict(data)
        data = None
        gc.collect()
        feauture_engg_data = pd.DataFrame(feauture_engg_data)
       
        return feauture_engg_data,label
    
    feauture_engg_data,label = featureExtract()
    
    np.save("FeatureExtractData_"+dataType+".npy",feauture_engg_data.values)
    np.save("FeatureExtractLabel_"+dataType+".npy",label)
    
    return "FeatureExtractData_"+dataType+".npy","FeatureExtractLabel_"+dataType+".npy"



#Select song for recommendation
def selectSong(dataType,songNumber):
    #Calculate distance matrix
    feauture_engg_data = np.load("FeatureExtractData_"+dataType+".npy",allow_pickle=True)
    label = np.load("FeatureExtractLabel_"+dataType+".npy",allow_pickle=True)
    distances = pdist(feauture_engg_data, metric='euclidean')
    dist_matrix = squareform(distances)
    
    sample = np.array(dist_matrix[songNumber][:])
    #Order distance to other songs
    order = np.argsort(sample)
    songName = label[order[0]]
    #Select near songs
    order = order[1:6]
    #Get recomended songs' name
    recommend = label[order]
    print("Song:",songName,"\nRecommends:\n", recommend)
