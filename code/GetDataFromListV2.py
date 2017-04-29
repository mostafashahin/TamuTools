from __future__ import print_function
import sys
import numpy as np
from sklearn.externals import joblib

def GetXy(lFeatMask,iAttributeIndx,iNumFrames,iFeatSize,bBalanceIt=False):
    arFeatData = np.zeros((1000000,iFeatSize),dtype='float32') #We can use empty instead of zeros to speed up but each element should be setted
    vLabel = np.zeros((1000000),dtype='int32')-1 #We can use empty instead of zeros to speed up
    iPaddingFrames = iNumFrames/2 #Number of Frames added to the start and end of each uttrance
    iPntr = iPaddingFrames
    #with open(sListOfFiles) as fListOfFiles:
    #    lLines = fListOfFiles.read().splitlines()
    iTotalFiles = len(lFeatMask)
    iCur = 0
    for tFeatMask in lFeatMask:
        print(int((float(iCur)/iTotalFiles)*100),'%',end='\r')
        sFeatFile,sMaskFile = tFeatMask
        arFileFeat = joblib.load(sFeatFile)
        arMask = joblib.load(sMaskFile)
        vMask = arMask[:,iAttributeIndx]
        iLenthFeat = arFileFeat.shape[0]
        iLenthLabel = vMask.shape[0]
        arFeatData[iPntr:iPntr+iLenthFeat] = arFileFeat
        vLabel[iPntr:iPntr+iLenthLabel] = vMask
        iPntr += max(iLenthFeat,iLenthLabel)+iPaddingFrames
        iCur += 1
    arFeatData = arFeatData[:iPntr]
    vLabel = vLabel[:iPntr]
    vPveInds = np.where(vLabel==1)[0]
    vNveInds = np.where(vLabel==0)[0]
    if bBalanceIt:
        iNumSelected = min(vPveInds.shape[0],vNveInds.shape[0])
        np.random.shuffle(vPveInds)
        np.random.shuffle(vNveInds)
        vPveInds = vPveInds[:iNumSelected]
        vNveInds = vNveInds[:iNumSelected]
    vSelectedIndx = np.r_[vPveInds,vNveInds]
    np.random.shuffle(vSelectedIndx)
    #arContextFeatData = np.empty((vSelectedIndx.shape[0],iNumFrames*iFeatSize),dtype='float32') #Could be used insted of concatenation to speed up
    lCotextData = [arFeatData[vSelectedIndx+i] for i in range(-iPaddingFrames,iPaddingFrames+1)]
    arContextFeatData = np.c_[tuple(lCotextData)]
    vSelectedLabel = vLabel[vSelectedIndx]
    return(arContextFeatData,vSelectedLabel)
    #arContextFeatData = arCurCotextFeat        
    #print(arFeatData.shape,vLabel.shape,vSelectedIndx.shape,arContextFeatData.shape)
    
    #print(arContextFeatData[0])
    #print(vSelectedLabel[:60])

def GetX(lFeat,iNumFrames,iFeatSize):
    arFeatData = np.zeros((1000000,iFeatSize),dtype='float32') #We can use empty instead of zeros to speed up but each element should be setted
    vSelectedIndx = np.zeros((1000000),dtype='bool')
    iPaddingFrames = iNumFrames/2 #Number of Frames added to the start and end of each uttrance
    iPntr = iPaddingFrames
    lFileStartIndx = []
    #with open(sListOfFiles) as fListOfFiles:
    #    lLines = fListOfFiles.read().splitlines()
    iTotalFiles = len(lFeat)
    iCur = 0
    lFileStartIndx.append(0)
    for sFeatFile in lFeat:
        print(int((float(iCur)/iTotalFiles)*100),'%',end='\r')
        #print('TotalFiles = ',iTotalFiles)
        #print(sFeatFile)
        arFileFeat = joblib.load(sFeatFile)
        iLenthFeat = arFileFeat.shape[0]
        #print('\niPntr = ',iPntr,' iLenthFeat = ',iLenthFeat,'\n')
        arFeatData[iPntr:iPntr+iLenthFeat] = arFileFeat
        vSelectedIndx[iPntr:iPntr+iLenthFeat] = 1
        iPntr += iLenthFeat+iPaddingFrames
        iCur += 1
        lFileStartIndx.append(lFileStartIndx[-1]+iLenthFeat)
    lFileStartIndx = lFileStartIndx[:-1]
    arFeatData = arFeatData[:iPntr]
    vSelectedIndx = vSelectedIndx[:iPntr]
    vSelectedIndx = np.where(vSelectedIndx)[0] 
    #arContextFeatData = np.empty((vSelectedIndx.shape[0],iNumFrames*iFeatSize),dtype='float32') #Could be used insted of concatenation to speed up
    lCotextData = [arFeatData[vSelectedIndx+i] for i in range(-iPaddingFrames,iPaddingFrames+1)]
    arContextFeatData = np.c_[tuple(lCotextData)]
    return (arContextFeatData,lFileStartIndx)
    #arContextFeatData = arCurCotextFeat        
    #print(arFeatData.shape,vLabel.shape,vSelectedIndx.shape,arContextFeatData.shape)

    #print(arContextFeatData[0])
    #print(vSelectedLabel[:60])
if __name__ == "__main__":
    sListOfFiles = sys.argv[1]
    iAttributeIndx = int(sys.argv[2])
    iNumFrames = int(sys.argv[3])
    featSize = int(sys.argv[4])
    with open(sListOfFiles) as fListOfFiles:
        lLines = fListOfFiles.read().splitlines()
        lFeatMaskFiles = [tuple(line.split()) for line in lLines]
        lFeat = [line.split()[0] for line in lLines]
    X = GetX(lFeat,iNumFrames,featSize)
    print (X.shape)
