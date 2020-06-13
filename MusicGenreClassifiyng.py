# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 15:39:22 2020

@author: VEFAUĞURBİLGE
"""
from createSpectrogram import createSpecs
from createChroma import createChromaCQT
from createDatabase import dataBaseCreate
from CNN1model import createCNN1model
from CNN2model import createCNN2model
from FeatureExtractAndML import trainCNNmodel,featureExtractAndML
from Recommend import recommend,selectSong
from recommend_success import calculateRecommendSuccess

def main():
    #Create spectrograms graph from genres songs
    createSpecs()
    #Create chromagrams graph from genres songs
    createChromaCQT()
    #Create databases from chromagrams and spectrograms graphs
    dataBaseCreate()
    #Create CNN model with dataType info as input
    createCNN1model("spectrograms")
    createCNN2model("spectrograms")
    #Retrain the best model 
    trainCNNmodel()
    #Feature extraction from best model and classify with ML algorithms
    featureExtractAndML()
    #Select a pickle file and create a new feature data set from it
    featere_data,label = recommend("turkish spectrograms")
    #With new feature dataset selecting a song number and recommendation songs
    selectSong("turkish spectrograms",14)
    #Calculate recommendation success
    calculateRecommendSuccess(featere_data,label)