# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import librosa, librosa.display
import IPython.display as ipd
import pylab
import os
import fnmatch

def createSpecs():
    def createSpectrogram6s(genreType, windowType, index):
        loadPath = 'genres/' + genreType + '/' + genreType + '.000' + index + '.au'
        fig = pylab.gcf()
        fig.set_size_inches(2.4, 2.4)
        pylab.axis('off')  # no axis
        pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])  # Remove the white edge
        #For 6 seconds
        for i in range(0, 30, 6):
            x, sr = librosa.load(loadPath, offset=i, duration=6.0)
            ipd.Audio(x, rate=sr)
            X = librosa.stft(x, window=windowType)
            S = librosa.amplitude_to_db(abs(X))
    
            librosa.display.specshow(S)
            savePath = "spectrograms6s"+'/' + genreType + '/' + genreType + '.000' + index + '_' + windowType + '_sec' + str(i) + '.png'
            pylab.savefig(savePath, bbox_inches=None, pad_inches=0)
        pylab.close()
    
    
    def createSpectrogram6sDataBase():
        genreTypes = list({'hiphop', 'blues', 'classical', 'country', 'disco', 'jazz', 'metal', 'pop', 'reggae', 'rock'})
        windowTypes = list({'hann', 'bohman', 'blackmanharris', 'nuttall', 'barthann'})
        indexList = list()
    
        for i in range(0, 100):
            if (i / 10) < 1:
                index = '0' + str(i)
            else:
                index = str(i)
            indexList.append(index)
    
        for genType in genreTypes:
            print(genType + "is started...")
            os.makedirs("spectrograms6s"+'/' + genType)
            for index in indexList:
                for windowType in windowTypes:
                    createSpectrogram6s(genType, windowType, index)
                    
    def createSpectrogram(genreType, windowType, index):
        loadPath = 'genres/' + genreType + '/' + genreType + '.000' + index + '.au'
    
        fig = pylab.gcf()
        fig.set_size_inches(2.4, 2.4)
        pylab.axis('off')  # no axis
        pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])  # Remove the white edge
        
        
        x, sr = librosa.load(loadPath)
        ipd.Audio(x, rate=sr)
        X = librosa.stft(x, window=windowType)
        S = librosa.amplitude_to_db(abs(X))
    
        librosa.display.specshow(S)
        savePath = "spectrograms"+'/' + genreType + '/' + genreType + '.000' + index + '_' + windowType +'.png'
        pylab.savefig(savePath, bbox_inches=None, pad_inches=0)
        pylab.close()
    
    
    def createSpectrogramDataBase(graphicType):
        genreTypes = list({'hiphop', 'blues', 'classical', 'country', 'disco', 'jazz', 'metal', 'pop', 'reggae', 'rock'})
        windowTypes = list({'hann', 'bohman', 'blackmanharris', 'nuttall', 'barthann'})
        indexList = list()
    
        for i in range(0, 100):
            if (i / 10) < 1:
                index = '0' + str(i)
            else:
                index = str(i)
            indexList.append(index)
    
        for genType in genreTypes:
            print(genType + "is started...")
            os.makedirs("spectrograms"+'/' + genType)
            for index in indexList:
                for windowType in windowTypes:
                    createSpectrogram(genType, windowType, index)
    
    
    createSpectrogram6sDataBase()
    createSpectrogramDataBase()

def createSpecForTurkish():

    def createTurkishSpectrogram(fileName):
        for root, dirnames, filenames in os.walk(fileName+'/'):
            for filename in fnmatch.filter(filenames, '*.wav'):
                musicPath = os.path.join(root, filename)
                musicName = musicPath.split('/')[-1]
                musicCategory = musicName.split('\\')[0]
                musicName = musicName.split('\\')[1]
                musicName = musicName.split('.')[0]
                fig = pylab.gcf()
                fig.set_size_inches(2.4, 2.4)
                pylab.axis('off')  # no axis
                pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])  # Remove the white edg
                x, sr = librosa.load(musicPath)
                ipd.Audio(x, rate=sr)
                X = librosa.stft(x)
                S = librosa.amplitude_to_db(abs(X))
                librosa.display.specshow(S)
                savePath = 'turkish spectrograms/' +"("+ musicCategory + ") " + musicName + '.png'
                pylab.savefig(savePath, bbox_inches=None, pad_inches=0)
            pylab.close()
    
    createTurkishSpectrogram("Music_30sn")