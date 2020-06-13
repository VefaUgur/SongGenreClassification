# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 16:25:21 2020

@author: VEFAUĞURBİLGE

tensorflow = 2.0.0
"""

import tensorflow as tf
tf.__version__

import pandas as pd 
import numpy as np
import gc
import seaborn as sns
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

from sklearn import model_selection, preprocessing, metrics
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import itertools
import tensorflow as tf
from tensorflow.python.keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,UpSampling2D
from tensorflow.python.keras.optimizers import RMSprop,Adam
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import ReduceLROnPlateau
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
import time
model_selection.KFold
from tqdm import tqdm
import concurrent.futures
import tensorflow as tf
from tensorflow.keras import backend as K
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier,AdaBoostClassifier
from sklearn import metrics
import pickle
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
    
import seaborn as sns
import tensorflow as tf 
from sklearn.metrics import classification_report

def apply():
    print("\nCreating model for feature extraction...")
    trainCNNmodel()
    print("Model was created")
    print("\nData is transforming and Classifiyng...")
    featureExtractAndML()
    



def trainCNNmodel(dataType = "spectrograms"):
    def getDataWithLabel():
      #%read Greyscale images
      data = pd.read_pickle('DataSets/'+dataType+"_data.pkl") 
      data = data.sample(frac=1).reset_index(drop=True)
      #%
      # put labels into y_train variable
      labels = data['Category']
      # Drop 'label' column
      data = data.drop(labels = ['Category'],axis = 1) 
      return data,labels
    
    def labelEncode(i):
      if  'blues' == i:
        return 0
      elif 'classical'== i:
        return 1
      elif 'country'== i:
        return 2
      elif 'disco'== i:
        return 3
      elif 'hiphop'== i:
        return 4
      elif 'jazz'== i:
        return 5
      elif 'metal'== i:
        return 6
      elif 'pop'== i:
        return 7
      elif 'reggae'== i:
        return 8
      else: 
        return 9
    def labelDecode(i):
      if  0 == i:
        return 'blues'
      elif 1== i:
        return "classical"
      elif 2== i:
        return "country"
      elif 3== i:
        return "disco"
      elif 4== i:
        return "hiphop"
      elif 5== i:
        return "jazz"
      elif 6== i:
        return "metal"
      elif 7== i:
        return "pop"
      elif 8 == i:
        return "reggae"
      else:
        return "rock"
    
    def fitLabelEncoder(labels):
      labelsEncode = []
      for i in range(labels.shape[0]):
        labelsEncode.append(labelEncode(labels[i]))
      labelsEncode = np.array(labelsEncode)
      return labelsEncode
  
    def fitLabelDecoder(labels):
      labelsDecode = []
      for i in range(labels.shape[0]):
        labelsDecode.append(labelDecode(labels[i]))
      labelsDecode = np.array(labelsDecode)
      return labelsDecode
    
    
    def createTestAndTrain():
      X_train,Y_train = getDataWithLabel()
      # Normalize the data
      X_train= X_train.astype('float16')
      X_train = X_train / 255.0
      print("Data was normalized..")
      print("Data shape: ",X_train.shape)
      #Reshape to matrix
      X_train = X_train.values.reshape(-1,240,240,3)
      print("Data was reshaped..")
      #LabelEncode
      #labels = preprocessing.LabelEncoder().fit_transform(labels)
      Y_train = fitLabelEncoder(Y_train)
      print("Data was encoded..")
      #int to vector
      Y_train = to_categorical(Y_train, num_classes = 10)
      #train and test data split
    
      X_train, X_test, Y_train, Y_test= train_test_split(X_train, Y_train, test_size=0.2, random_state=42)
      return X_train, X_test, Y_train, Y_test;
      
    
    def createModel(X_train):
      model = Sequential()
      #
      model.add(Conv2D(filters = 16, kernel_size = (3,3),padding = 'Same', 
                      activation ='relu', input_shape = (X_train.shape[1],X_train.shape[2],X_train.shape[3])))
      model.add(MaxPool2D(pool_size=(2,2)))
      model.add(Dropout(0.25))
      #
      model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', 
                      activation ='relu'))
      model.add(MaxPool2D(pool_size=(2,2)))
      model.add(Dropout(0.25))
        ##decode
      model.add(Conv2D(filters = 16, strides=(2,2), kernel_size = (3,3),padding = 'Same', 
                      activation ='relu'))
      #model.add(UpSampling2D((2,2)))
      model.add(Conv2D(filters = 16, kernel_size = (3,3),padding = 'Same', 
                      activation ='relu'))
      model.add(UpSampling2D((2,2)))
      ##
      # fully connected
      model.add(Flatten())
      model.add(Dense(256, activation = "relu"))
      model.add(Dropout(0.5))
      model.add(Dense(10, activation = "softmax"))
      #%
      # Define the optimizer
      optimizer = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
      #%
      model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
      model.summary()
      return model
    
      #set early stopping criteria
    pat = 10 #this is the number of epochs with no improvment after which the training will stop
    early_stopping = EarlyStopping(monitor='val_loss', patience=pat, verbose=1)
    
    #define the model checkpoint callback -> this will keep on saving the model as a physical file
    checkpointPath = dataType+'_CNN2_feature_checkpoint.h5'
    model_checkpoint = ModelCheckpoint(checkpointPath, verbose=1, save_best_only=True)
    
    def plotCategories(y_train,val_y):
      Y_train_classes = np.argmax(y_train,axis = 1) 
      Y_train_classes = fitLabelDecoder(Y_train_classes)
      
      plt.figure(figsize=(15,7)) 
      g = sns.countplot(Y_train_classes, palette="icefire")
      plt.title("Train Number of digit classes")
      plt.show()
      
      Y_val_classes = np.argmax(val_y,axis = 1) 
      Y_val_classes = fitLabelDecoder(Y_val_classes)
      plt.figure(figsize=(15,7))
      g = sns.countplot(Y_val_classes, palette="icefire")
      plt.title("Validation Number of digit classes")
      plt.show()
      gc.collect()
    
    def fit_and_evaluate(X_train,y_train):
      model = None
      gc.collect()
      model = createModel(X_train)
      batch_size = 32
      epochs = 30
      gc.collect()
      datagen = ImageDataGenerator(zoom_range = 0.2,horizontal_flip = False)
      datagen.fit(X_train)
      gc.collect()
      train_x, val_x, train_y, val_y = train_test_split(X_train, y_train, test_size=0.1, random_state = np.random.randint(1,1000, 1)[0])
      plotCategories(train_y,val_y)
      results = model.fit_generator(datagen.flow(train_x,train_y,batch_size=batch_size), epochs = epochs,steps_per_epoch = X_train.shape[0] // batch_size ,callbacks=[early_stopping, model_checkpoint], 
                verbose=1,validation_data = (val_x,val_y))  
      gc.collect()
      print("Val Score: ", model.evaluate(val_x, val_y))
      return 
    
    X_train, X_test, Y_train, Y_test = createTestAndTrain()
    np.save("X_train.npy",X_train)
    np.save("Y_train.npy",Y_train)
    np.save("X_test.npy",X_test)
    np.save("Y_test.npy",Y_test)
    gc.collect()
    print("Train and Test Data Saved.")
          
    fit_and_evaluate(X_train,Y_train)
    gc.collect()
    
    def predictTest(X_test, Y_test,lmodel):
      Y_pred = lmodel.predict(X_test)
      # Convert predictions classes to one hot vectors 
      Y_pred_classes = np.argmax(Y_pred,axis = 1) 
      Y_pred_classes = fitLabelDecoder(Y_pred_classes)
      Y_test_label = np.argmax(Y_test,axis = 1) 
      Y_test_label = fitLabelDecoder(Y_test_label)
      # compute the confusion matrix 
      labels=["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"]
      confusion_mtx = confusion_matrix(Y_test_label, Y_pred_classes) 
      #plot the confusion matrix
      f,ax = plt.subplots(figsize=(8, 8))
      sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="Greens",linecolor="gray", fmt= '.1f',ax=ax, xticklabels=labels ,yticklabels=labels)
      plt.yticks(rotation=0)
      plt.xlabel("Predicted Label")
      plt.ylabel("True Label")
      plt.title("Confusion Matrix")
      plt.show()
      acc = metrics.accuracy_score(Y_test_label, Y_pred_classes)*100
      print('Accuracy percentage:', acc)
      
      print(classification_report(Y_test_label, Y_pred_classes)) 
      #score = lmodel.evaluate(X_test, Y_test, verbose=0)
      #print('Test loss:', score[0])
      #print('Test accuracy:', score[1])
      return
    
    lmodel = tf.keras.models.load_model(dataType+'_CNN2_feature_checkpoint.h5')
    predictTest(X_test, Y_test,lmodel)

#Feature Extraction From Saved Model and Classify With ML Algortihms
def featureExtractAndML():
    
    model = tf.keras.models.load_model('spectrograms_CNN2_feature_checkpoint.h5')
    intermediate_layer_model = tf.keras.Model(inputs=model.input,outputs=model.get_layer('dropout_2').output)
    intermediate_layer_model.summary()
    
    #Load train test data
    X_train = np.load("X_train.npy",allow_pickle=True) 
    X_test = np.load("X_test.npy",allow_pickle=True) 
    Y_train = np.load("Y_train.npy") 
    Y_test = np.load("Y_test.npy") 
    print("Train test data loaded.")
    
    #Feature Extraction create new dataset
    feauture_X_train = intermediate_layer_model.predict(X_train)
    feauture_X_test = intermediate_layer_model.predict(X_test)
    gc.collect()
    
    #np.save("Feature_X_train.npy",feauture_X_train)
    #np.save("Feature_X_test.npy",feauture_X_test)
    
    print('feauture_engg_data shape:', feauture_X_train.shape)
    print('feauture_engg_data shape:', feauture_X_test.shape)
    
    feature_Y_train = np.argmax(Y_train,axis = 1) 
    feature_Y_test = np.argmax(Y_test,axis = 1) 
    
    def labelDecode(i):
          if  0 == i:
            return 'blues'
          elif 1== i:
            return "classical"
          elif 2== i:
            return "country"
          elif 3== i:
            return "disco"
          elif 4== i:
            return "hiphop"
          elif 5== i:
            return "jazz"
          elif 6== i:
            return "metal"
          elif 7== i:
            return "pop"
          elif 8 == i:
            return "reggae"
          else:
            return "rock"
        
    def fitLabelDecoder(labels):
        labelsDecode = []
        for i in range(labels.shape[0]):
          labelsDecode.append(labelDecode(labels[i]))
        labelsDecode = np.array(labelsDecode)
        return labelsDecode
    
    def getResults(Y_pred_classes,y_test):
      Y_pred_classes = fitLabelDecoder(Y_pred_classes)
      Y_test_label = fitLabelDecoder(y_test)
      # compute the confusion matrix 
      labels=["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"]
      confusion_mtx = confusion_matrix(Y_test_label, Y_pred_classes) 
      # plot the confusion matrix
      f,ax = plt.subplots(figsize=(8, 8))
      sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="Greens",linecolor="gray", fmt= '.1f',ax=ax, xticklabels=labels ,yticklabels=labels)
      plt.yticks(rotation=0)
      plt.xlabel("Predicted Label")
      plt.ylabel("True Label")
      plt.title("Confusion Matrix")
      plt.show()
    
      print(classification_report(Y_test_label, Y_pred_classes))
      
      #BestModel
    random_state = 42
    models=[MLPClassifier(max_iter=2000, activation='tanh',solver='lbfgs', random_state=random_state),
           LogisticRegression(dual=False,multi_class='auto',solver='lbfgs',random_state=random_state),
           RandomForestClassifier(n_estimators=100,criterion='entropy'),
           LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto'),
           KNeighborsClassifier(n_neighbors=10, weights='distance'),
           SVC(kernel='poly',degree=2,C=100, gamma='auto'),
           GaussianNB(),
           GradientBoostingClassifier(),
           AdaBoostClassifier(),
           LinearSVC(penalty='l1',dual=False,multi_class='crammer_singer',max_iter=1000000),
           SVC(kernel='rbf', random_state=0, gamma=.01, C=100000)]
    
    def best_model(X_train,y_train,X_test,y_test,show_metrics = True):
      print("---------------")
      print("INFO: Finding Accuracy Best Classifier...", end="\n\n")
      best_clf=None
      best_acc=0
      best_model = None
      for clf in models:
          clf.fit(X_train, y_train)
          y_pred=clf.predict(X_test)
          acc=metrics.accuracy_score(y_test, y_pred)
          print(clf.__class__.__name__, end=" ")
          print("Accuracy:{:.3f}".format(acc))
          precision = precision_score(y_test, y_pred,average='macro')
          recall = recall_score(y_test, y_pred,average='macro')
          f1 = f1_score(y_test, y_pred,average='macro')
          print("*Pre:{:.3f}".format(precision)," *Rec:{:.3f}".format(recall)," *F1:{:.3f}".format(f1))
          print("---------------")
         # print(classification_report(y_test, y_pred))
          
    
          if best_acc<acc:
              best_model = clf
              best_acc=acc
              best_clf=clf
              best_y_pred=y_pred
          
          filename = 'finalized_classifyng_model(30sec).sav'
          pickle.dump(best_model, open(filename, 'wb'))
    
      print("Best Classifier:{}".format(best_clf.__class__.__name__))
      if show_metrics:
        getResults(best_y_pred,y_test)
        
    best_model(feauture_X_train,feature_Y_train,feauture_X_test,feature_Y_test)




