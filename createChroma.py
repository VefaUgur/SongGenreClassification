import librosa, librosa.display
import pylab
import os

def createChromaCQT():
    def createChroma6s(genreType, index):
        loadPath = 'genres/' + genreType + '/' + genreType + '.000' + index + '.au'
    #    savePath = 'chorama_cqt/' + genreType + '/' + genreType + '.000' + index + '_' + windowType + '.png'
        savePath = 'chroma6s/' + genreType + '/' + genreType + '.000' + index +'.png'
    
        fig = pylab.gcf()
        fig.set_size_inches(2.4, 2.4)
        # for 6 sec
        i = 0
        pylab.axis('off')  # no axis
        pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])  # Remove the white edg
        for i in range(0,30,6):
            y, sr = librosa.load(loadPath, offset = i, duration = 6.0)
            chroma_cq = librosa.feature.chroma_cqt(y=y, sr=sr)
            librosa.display.specshow(chroma_cq)
            savePath = 'chroma6s/' + genreType + '/' + genreType + '.000' + index +'_sec'+ str(i) +'.png'
            pylab.savefig(savePath, bbox_inches=None, pad_inches=0)
        pylab.close()
    
    
    def createChromaDataBase6s():
        genreTypes = list({'blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock'})
        indexList = list()
    
        for i in range(0, 100):
            if (i / 10) < 1:
                index = '0' + str(i)
            else:
                index = str(i)
            indexList.append(index)
    
        for genType in genreTypes:
            print(genType + " is started...")
            os.makedirs('chroma6s/' + genType)
            for index in indexList:
                createChroma6s(genType, index)
    #for windowType in windowTypes:
                
    def createChroma(genreType, index):
        loadPath = 'genres/' + genreType + '/' + genreType + '.000' + index + '.au'
    #    savePath = 'chorama_cqt/' + genreType + '/' + genreType + '.000' + index + '_' + windowType + '.png'
        savePath = 'chroma/' + genreType + '/' + genreType + '.000' + index +'.png'
    
        fig = pylab.gcf()
        fig.set_size_inches(2.4, 2.4)
        # for 6 sec
        pylab.axis('off')  # no axis
        pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])  # Remove the white edg
      
        y, sr = librosa.load(loadPath)
        chroma_cq = librosa.feature.chroma_cqt(y=y, sr=sr)
        librosa.display.specshow(chroma_cq)
        savePath = 'chroma/' + genreType + '/' + genreType + '.000' + index +'.png'
        pylab.savefig(savePath, bbox_inches=None, pad_inches=0)
        pylab.close()
    
    
    def createChromaDataBase():
        genreTypes = list({'blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock'})
        indexList = list()
    
        for i in range(0, 100):
            if (i / 10) < 1:
                index = '0' + str(i)
            else:
                index = str(i)
            indexList.append(index)
    
        for genType in genreTypes:
            print(genType + " is started...")
            os.makedirs('chroma/' + genType)
            for index in indexList:
                createChroma(genType, index)
    #for windowType in windowTypes:
    
    createChromaDataBase6s()
    createChromaDataBase()
