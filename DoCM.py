from __future__ import print_function
from os.path import basename, splitext, join
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
import sys,glob
sys.path.append('/panfs/vol/m/moshahi34/TamuTools/code')
import cPickle
from GetDataFromListV2 import GetX
from DNN_DecodeV2 import Decode

iSampleRate = 10
def GetFeaturData(sFileList, sFeatPath):
    with open(sFileList) as fList:
        lChunkFiles = [join(sFeatPath,splitext(basename(sFile))[0]+'.feat') for sFile in fList.read().splitlines()]
    arFeatData,lStartIndxs = GetX(lChunkFiles,9,78)
    dFileIndx = {}
    for i in range(len(lChunkFiles)):
        sFile = lChunkFiles[i]
        dFileIndx[splitext(basename(sFile))[0]] = lStartIndxs[i]
    return arFeatData,dFileIndx

def loadTrainingData(sPhone,sDataPath,iNumAttributes):
    lFiles = glob.glob(join(sDataPath,sPhone+'_Train_*'))
    arPhoneData = np.empty((3000000,iNumAttributes),dtype='float')
    iPoint = 0
    for sFile in lFiles:
        if splitext(sFile)[1] == '':
            #print(sFile)
            arData = joblib.load(sFile)
            iCurSize = arData.shape[0]
            arPhoneData[iPoint:iPoint+iCurSize] = arData
            iPoint +=iCurSize
    return arPhoneData[:iPoint]

def EvaluatePhone(sMlfFile, sPhoneAttrFile,sFileList):
    with open(sFileList) as fList:
        lFiles = [splitext(basename(sFile))[0] for sFile in fList.read().splitlines()]
    dPhonAttr = {}
    with open(sPhoneAttrFile) as fPhAtt:
        for sLine in fPhAtt.read().splitlines():
            lLineS = sLine.split()
            dPhonAttr[lLineS[0]] = lLineS[1:]
    with open(sMlfFile) as fMlf:
        lMlf = fMlf.read().splitlines()
    lSEInx = [i for i in range(len(lMlf)) if lMlf[i][0]=='"' or lMlf[i][0] == '.'] #Get Start and End index of each file
    dPhoneScalModel = {}
    dFileEval = {}
    for i in range(0,len(lSEInx),2):
        print(int((float(i)/len(lSEInx))*100),'%',end='\r')
        iSinx = lSEInx[i]
        iEinx = lSEInx[i+1]
        sFileName = splitext(basename(lMlf[iSinx][1:-1]))[0]
        if sFileName not in lFiles:
            continue
        dFileEval[sFileName] = []
        arA = np.asarray([])
        arMT = np.asarray([])
        arBN = np.asarray([])
        arMT10 = np.asarray([])
        for j in range(iSinx+1,iEinx):
            lLine = lMlf[j].split()
            iSFram = int(lLine[0])/iSampleRate**5
            iEFram = int(lLine[1])/iSampleRate**5
            sPhoneme = lLine[2]
            if sPhoneme not in dPhonAttr:
                sAttributeFile = 'Attributes_params'
                sSufixModel = 'ASNSF'
                iBNFeatSize = -1
                sDataPath = 'phoneAttributes_fixed'
            else:
                sAttributeFile, sSufixModel, iBNFeatSize, sDataPath = dPhonAttr[sPhoneme]
                iBNFeatSize = int(iBNFeatSize)
            if iBNFeatSize == -1:
                iNumAttributes = 26 
            else:
                iNumAttributes = 26 * iBNFeatSize
            if sPhoneme not in dPhoneScalModel:
                sModel = join('results',sPhoneme+'_'+sSufixModel+'.model')
                #print(sModel)
                with open(sModel) as fModel:
                    clsf = cPickle.load(fModel)
                with open(join('results',sPhoneme+'_'+sSufixModel+'.log'))as fPhoneLog:
                    Line = fPhoneLog.read().splitlines()[-1]
                sLine = Line.split()
                if Line.find(',') == -1:
                    indx = [int(s.replace('[','').replace(',','').replace(']','')) for s in sLine[7:]]
                else:
                    indx = [int(s.replace('[','').replace(',','').replace(']','')) for s in sLine[6:]]
                arTraining = loadTrainingData(sPhoneme,sDataPath,iNumAttributes)#joblib.load(sTargetPhone+'.jbl')
                scal = StandardScaler()
                scal.fit(arTraining)
                dPhoneScalModel[sPhoneme] = (clsf,indx,scal)
            if sSufixModel == 'ASNSF':
                if arA.shape[0] == 0:
                    arA = joblib.load(join('featFiles_A',sFileName+'.feat'))
                arAttFeat = arA[iSFram:iEFram]
            elif sSufixModel == 'BN10F':
                if arBN.shape[0] == 0:
                    arBN = joblib.load(join('featFiles_BN',sFileName+'.feat'))
                arAttFeat = arBN[iSFram:iEFram]
            elif sSufixModel == 'MT':
                if arMT.shape[0] == 0:
                    arMT = joblib.load(join('featFiles_MT',sFileName+'.feat'))
                arAttFeat = arMT[iSFram:iEFram]
            elif sSufixModel == 'MT10':
                if arMT10.shape[0] == 0:
                    arMT10 = joblib.load(join('featFiles_MT10',sFileName+'.feat'))
                arAttFeat = arMT10[iSFram:iEFram]
            else:
                if arA.shape[0] == 0:
                    arA = joblib.load(join('featFiles_A',sFileName+'.feat'))
                arAttFeat = arA[iSFram:iEFram]
            clsf,indx,scal = dPhoneScalModel[sPhoneme]
            arAttFeat_std = scal.transform(arAttFeat)
            y_predict = clsf.predict(arAttFeat_std[:,indx])
            iNumInClassFrams = np.where(y_predict == 1)[0].shape[0]
            iNumOutClassFrams = np.where(y_predict == -1)[0].shape[0]
            if iNumInClassFrams >= 0.25*iNumOutClassFrams:
                _class = 1
            else:
                _class = 0
            dFileEval[sFileName].append((sPhoneme,_class))
    return dFileEval



def EvaluatePhone_(sMlfFile, sPhoneAttrFile, arFeatData, dFileIndx):
    dPhonAttr = {}
    with open(sPhoneAttrFile) as fPhAtt:
        for sLine in fPhAtt.read().splitlines():
            lLineS = sLine.split()
            dPhonAttr[lLineS[0]] = lLineS[1:]
    with open(sMlfFile) as fMlf:
        lMlf = fMlf.read().splitlines()
    lSEInx = [i for i in range(len(lMlf)) if lMlf[i][0]=='"' or lMlf[i][0] == '.'] #Get Start and End index of each file
    dPhones = {}
    for i in range(0,len(lSEInx),2):
        print(int((float(i)/len(lSEInx))*100),'%',end='\r')
        iSinx = lSEInx[i]  
        iEinx = lSEInx[i+1]
        sFileName = splitext(basename(lMlf[iSinx][1:-1]))[0]
        if sFileName not in dFileIndx:
            continue
        iPhIndx = 0
        for j in range(iSinx+1,iEinx):
            lLine = lMlf[j].split()
            iSFram = int(lLine[0])/iSampleRate**5
            iEFram = int(lLine[1])/iSampleRate**5
            sPhoneme = lLine[2]
            if sPhoneme not in dPhones:
                dPhones[sPhoneme] = []
            dPhones[sPhoneme].append((sFileName,iSFram,iEFram,iPhIndx))
            iPhIndx += 1
    for sPhone in dPhones:
        arData = np.empty((arFeatData.shape),dtype=float)
        iPntr = 0
        for i in range(len(dPhones[sPhone])):
            sFileName,iSFram,iEFram,iPhIndx = dPhones[sPhone][i]
            dPhones[sPhone][i] = dPhones[sPhone][i] + (iPntr,)
            iSIndx = dFileIndx[sFileName] + iSFram
            iEIndx = dFileIndx[sFileName] + iEFram
            arData[iPntr:iPntr+(iEIndx-iSIndx)] = arFeatData[iSIndx:iEIndx]
            iPntr+=(iEIndx-iSIndx)
        arData = arData[:iPntr]
        if sPhone not in dPhonAttr:
            sAttributeFile = 'Attributes_params'
            sSufixModel = 'ASNSF'
            iBNFeatSize = -1
            sDataPath = 'phoneAttributes_fixed'
        else:
            sAttributeFile, sSufixModel, iBNFeatSize, sDataPath = dPhonAttr[sPhone]
            iBNFeatSize = int(iBNFeatSize) 
        with open(sAttributeFile) as fAttribParams:
            lAttribParams = [[int(i) for i in sLine.split()[:-1]]+[sLine.split()[-1]] for sLine in fAttribParams.read().splitlines()]
            #print(lAttribParams)
        if iBNFeatSize == -1:
            arAttributeFeatures = np.empty((arData.shape[0],len(lAttribParams)),dtype='float')
            iNumAttributes = len(lAttribParams)
        else:
            arAttributeFeatures = np.empty((arData.shape[0],len(lAttribParams)*iBNFeatSize),dtype='float')
            iNumAttributes = len(lAttribParams) * iBNFeatSize
        print(arData[0],arData[1])
        
        for iParam in range(len(lAttribParams)):
            print('********************',iParam)
            iAttributeIndx,iNFrams,iNhl,iNhu,iBNFeatSize,sBestParamFile = lAttribParams[iParam]
            Py, yP = Decode(iNhu, iNhl, 2, arData, sBestParamFile,iBNhl=3,iBNhu=iBNFeatSize)
            print(Py)
            if iBNFeatSize == -1:
                arAttributeFeatures[:,iParam] = Py[:,0]
            else:
                arAttributeFeatures[:,iParam*iBNFeatSize:iParam*iBNFeatSize+iBNFeatSize] = Py
        arFeat = arAttributeFeatures
        for i in range(len(dPhones[sPhone])):
            item = dPhones[sPhone][i]
            iPntr = item[-1]
            iLength = item[2]-item[1]
            arAttFeat = arFeat[iPntr:iPntr+iLength]
            sModel = join('results',sPhone+'_'+sSufixModel+'.model')
            print(sModel)
            with open(sModel) as fModel:
                clsf = cPickle.load(fModel)
            #scal = StandardScaler()
            print(arAttFeat)
            with open(join('results',sPhone+'_'+sSufixModel+'.log'))as fPhoneLog:
                Line = fPhoneLog.read().splitlines()[-1]
            sLine = Line.split()
            sKernel = sLine[5]
            sNu = sLine [3]
            sGamma = sLine[4]
            if Line.find(',') == -1:
                indx = [int(s.replace('[','').replace(',','').replace(']','')) for s in sLine[7:]]
            else:
                indx = [int(s.replace('[','').replace(',','').replace(']','')) for s in sLine[6:]]
            arTraining = loadTrainingData(sPhone,sDataPath,iNumAttributes)#joblib.load(sTargetPhone+'.jbl')
            scal = StandardScaler()
            arTraining_std = scal.fit_transform(arTraining)[:,indx]
            arAttFeat_std = scal.transform(arAttFeat)
            y_predict = clsf.predict(arAttFeat_std[:,indx])
            iNumInClassFrams = np.where(y_predict == 1)[0].shape[0]
            iNumOutClassFrams = np.where(y_predict == -1)[0].shape[0]
            if iNumInClassFrams >= 0.5*iNumOutClassFrams:
                _class = 1
            else:
                _class = 0
            dPhones[sPhone][i] = dPhones[sPhone][i] + (_class,)
        print(arFeat.shape)
        print(sPhone,arData.shape,sAttributeFile, sSufixModel)
    return(dPhones)

def ComputeAcc(dFileEval,sRefMLF):
    with open(sRefMLF) as fMlf:
        lMlf = fMlf.read().splitlines()
    lSEInx = [i for i in range(len(lMlf)) if lMlf[i][0]=='"' or lMlf[i][0] == '.'] #Get Start and End index of each file
    fTotal = 0.0
    iCA = 0
    iCR = 0
    iFA = 0
    iFR = 0
    dPhone = {}
    for i in range(0,len(lSEInx),2):
        print(int((float(i)/len(lSEInx))*100),'%',end='\r')
        iSinx = lSEInx[i]
        iEinx = lSEInx[i+1]
        sFileName = splitext(basename(lMlf[iSinx][1:-1]))[0]
        if sFileName not in dFileEval:
            continue
        lEval = dFileEval[sFileName]
        iIndx = 0
        for line in lMlf[iSinx+1:iEinx]:
            sPhone = line.split('-')[-1]
            if sPhone not in dPhone:
                dPhone[sPhone] = [0,0,0,0]
            if '-' in line:
                if lEval[iIndx][1] == 0:
                    dPhone[sPhone][1] += 1
                    iCR += 1
                    fTotal += 1
                else:
                    dPhone[sPhone][2] += 1
                    iFA += 1
                    fTotal += 1
            else:
                if lEval[iIndx][1] == 0:
                    dPhone[sPhone][3] += 1
                    iFR += 1
                    fTotal += 1
                else:
                    dPhone[sPhone][0] += 1
                    iCA += 1
                    fTotal += 1
            iIndx += 1
    print(iCA,iCR,iFA,iFR)
    SA = ((iCA+iCR)/fTotal) * 100.0
    P_CA = (float(iCA)/(iCA+iFA)) * 100.0
    P_CR = (float(iCR)/(iCR+iFR)) * 100.0
    R_CA = (float(iCA)/(iCA+iFR)) * 100.0
    R_CR = (float(iCR)/(iCR+iFA)) * 100.0
    FA = 2*(P_CA*R_CA)/(P_CA+R_CA)
    print(dPhone)
    return(SA,P_CA,P_CR,R_CA,R_CR,FA)
    #return(dFileEval)
