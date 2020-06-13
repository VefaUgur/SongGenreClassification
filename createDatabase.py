import fnmatch
import os
import numpy as np
from PIL import Image
import pandas as pd

def dataBaseCreate():
    def createDatabasePickle(featureType):
        y_values = []
        x_values = []
        for root, dirnames, filenames in os.walk(featureType+'/'):
            for filename in fnmatch.filter(filenames, '*.png'):
                musicPath = os.path.join(root, filename)
                musicName = musicPath.split('/')[-1]
                musicCategory = musicName.split('\\')[0]         
               # musicCategory = musicName.split(')')[0]
                y_values.append(musicCategory)
                im = Image.open(musicPath).convert('RGB')
                
                x_values.append(np.array(im).flatten())
                
        x_values = np.array(x_values)
        dataframe = pd.DataFrame(x_values)
        dataframe['Category'] = y_values 
        dataframe.to_pickle("./"+featureType+"_data.pkl") 
    
    createDatabasePickle("chroma")
    createDatabasePickle("spectrograms")
    createDatabasePickle("chroma6s")
    createDatabasePickle("spectrograms6s")



