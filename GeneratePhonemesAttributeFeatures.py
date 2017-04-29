#!/usr/bin/env python
from __future__ import print_function
import sys,os
import gzip,cPickle
from collections import OrderedDict
import numpy as np
#import hickle as hkl
from random import shuffle
from sklearn.externals import joblib
sys.path.append('/panfs/vol/m/moshahi34/Tools/code')
from GetDataFromListV2 import GetX
from DNN_DecodeV2 import Decode
if len(sys.argv) != 9:
    print('Usage: GeneratePhonemesAttributeFeatures.py MlfFile ListAttributeParameters featFilePath TrainingList ValidList TestingList PhoneList FeatType')
    sys.exit(1)

sMLFFile = sys.argv[1]
sAttributeParam = sys.argv[2]
sFeaturePath = sys.argv[3]
fLogFile = open('GeneratePhonemesAttributeFeatures.log','w')
sTrainingScp,sValidScp,sTestScp = sys.argv[4:7]
sFeatType = sys.argv[8]
with open(sTrainingScp) as fTrainingScp, open(sValidScp) as fValidScp, open(sTestScp) as fTestScp:
    lTrainingFiles = [os.path.splitext(os.path.basename(sLine))[0] for sLine in fTrainingScp.read().splitlines()]
    lValidFiles = [os.path.splitext(os.path.basename(sLine))[0] for sLine in fValidScp.read().splitlines()]
    lTestFiles = [os.path.splitext(os.path.basename(sLine))[0] for sLine in fTestScp.read().splitlines()]
with open(sAttributeParam) as fAttribParams:
    lAttribParams = [[int(i) for i in sLine.split()[:-1]]+[sLine.split()[-1]] for sLine in fAttribParams.read().splitlines()]
    print(lAttribParams[0])
iMaxNumFrams = max([p[1] for p in lAttribParams])
#Configures
iSampleRate = 10 #Sampling rate of the feature files in mili seconds
sSavingFormat = 'J'
iNumChunkFiles = 300

#lPhonemes = ['sil','ao','aa','iy','uw','eh','ih','uh','ah','ax','ae','ey','ay','ow','aw','oy','er','axr','p','b','t','d','k','g','ch','jh','f','v','th','dh','s','z','sh','zh','hh','m','n','em','en','ng','eng','l','el','r','dx','nx','y','w','q']
with open(sys.argv[7]) as fPhonemes:
    lPhonemes = fPhonemes.read().splitlines()

#lAttributes = ['vowels','affricates','fricatives','nasals','liquids','semivowels','coronal','high','glottal','labial','low','mid','velar','retroflex','anterior','round']
#lAttributes = ['silience','vowels','stops','affricates','fricatives','nasals','liquids','semivowels','voiced','approx','coronal','high','dental','glottal','labial','low','mid','velar','retroflex','anterior','back','continuant','round','tense','Monophthongs','Diphthongs']
#with open(sys.argv[7]) as fAttributes:
#   dAttributes = OrderedDict([tuple([i[0],i[1:]]) for i in [sLine.split() for sLine in fAttributes.read().splitlines()]])

#enumAttributes = list(enumerate(lAttributes))
#Load Best Params List
#iNumAttributes = len(enumAttributes)
with open(sys.argv[2]) as fList:
    lListFiles = [os.path.splitext(os.path.basename(sLine))[0] for sLine in fList.read().splitlines()]
iChunkIndx = 0
with open(sys.argv[1]) as fMlf:
    iCounter=0
    lChunkTypes = []
    lChunkFiles = []
    dChunkPhone = {}
    for sPhone in lPhonemes:
        dChunkPhone[sPhone] = []
    lMlf = fMlf.read().splitlines()
    lSEInx = [i for i in range(len(lMlf)) if lMlf[i][0]=='"' or lMlf[i][0] == '.'] #Get Start and End index of each file
    for i in range(0,len(lSEInx),2):
        print(int((float(i)/len(lSEInx))*100),'%',end='\r')
        iSinx = lSEInx[i]
        iEinx = lSEInx[i+1]
        sFileName = os.path.splitext(os.path.basename(lMlf[iSinx][1:-1]))[0]
        if sFileName in lTrainingFiles:
            iFileType = 0 #Training
        elif sFileName in lValidFiles:
            iFileType = 1 # Valid
        elif sFileName in lTestFiles:
            iFileType = 2 #Test
        else:
            print('File: ',sFileName,' Not Registered',file=fLogFile)
            continue
        lChunkFiles.append(os.path.join(sFeaturePath,sFileName)+'.feat')
        lChunkTypes.append(iFileType)
        for j in range(iSinx+1,iEinx):
            lLine = lMlf[j].split()
            iSFram = int(lLine[0])/iSampleRate**5
            iEFram = int(lLine[1])/iSampleRate**5
            sPhoneme = lLine[2]
            dChunkPhone[sPhoneme].append([iCounter,iSFram,iEFram])
        iCounter += 1
        if iCounter == iNumChunkFiles:
            print('\n********************************iCounter = ',iCounter,len(lChunkFiles),'\n')
            arFeatData,lStartIndxs = GetX(lChunkFiles,iMaxNumFrams,78)
            #print(iChunkIndx,arFeatData.shape)

            if sFeatType == 'A': #Attribute Features
                arAttributeFeatures = np.empty((arFeatData.shape[0],len(lAttribParams)),dtype='float')
                for iParam in range(len(lAttribParams)):
                    print('********************',iParam)
                    iAttributeIndx,iNFrams,iNhl,iNhu,sBestParamFile = lAttribParams[iParam]
                    Py, yP = Decode(iNhu, iNhl, 2, arFeatData, os.path.join('Temp',sBestParamFile))
                    arAttributeFeatures[:,iParam] = Py[:,0]
                arFeat = arAttributeFeatures
            elif sFeatType == 'F':
                arFeat = arFeatData[:,312:390]
            else:
                arAttributeFeatures = np.empty((arFeatData.shape[0],len(lAttribParams)),dtype='float')
                for iParam in range(len(lAttribParams)):
                    iAttributeIndx,iNFrams,iNhl,iNhu,sBestParamFile = lAttribParams[iParam]
                    Py, yP = Decode(iNhu, iNhl, 2, arFeatData, os.path.join('Temp',sBestParamFile))
                    arAttributeFeatures[:,iParam] = Py[:,0]
                #arFeat = arAttributeFeatures
                arFeat = np.c_[arAttributeFeatures,arFeatData[:,312:390]]
                #arFeat = arFeatData
            for sPhone in lPhonemes:
                lPhoneTest = []
                vPhoneMask_Train = np.zeros((arFeatData.shape[0]),dtype='bool')
                vPhoneMask_Valid = np.zeros((arFeatData.shape[0]),dtype='bool')
                #vPhoneMask_Test = np.zeros((arFeatData.shape[0]),dtype='bool')
                for iFileIdx, iPSFram, iPEFram in dChunkPhone[sPhone]:
                    vPhoneMask_Test = np.zeros((arFeatData.shape[0]),dtype='bool')
                    iStartIndx = lStartIndxs[iFileIdx]
                    if lChunkTypes[iFileIdx] == 0:
                        #print(iStartIndx,iPSFram,iPEFram)
                        vPhoneMask_Train[iStartIndx+iPSFram: iStartIndx+iPEFram] = True
                    elif lChunkTypes[iFileIdx] == 1:
                        vPhoneMask_Valid[iStartIndx+iPSFram: iStartIndx+iPEFram] = True
                    else:
                        vPhoneMask_Test[iStartIndx+iPSFram: iStartIndx+iPEFram] = True
                        lPhoneTest.append(arFeat[vPhoneMask_Test])
                joblib.dump(arFeat[vPhoneMask_Train],sPhone+'_Train_'+sFeatType+'_'+str(iChunkIndx))
                joblib.dump(arFeat[vPhoneMask_Valid],sPhone+'_Valid_'+sFeatType+'_'+str(iChunkIndx))
                #joblib.dump(arFeat[vPhoneMask_Test],sPhone+'_Test_'+str(iChunkIndx))
                joblib.dump(lPhoneTest,sPhone+'_Test_'+sFeatType+'_'+str(iChunkIndx))
            iCounter=0
            lChunkTypes = []
            lChunkFiles = []
            #dChunkPhone = {}
            for sPhone in lPhonemes:
                #print(sPhone,len(dChunkPhone[sPhone]))#,dChunkPhone[sPhone][0])
                dChunkPhone[sPhone] = []
            iChunkIndx += 1
print(iCounter)
arFeatData,lStartIndxs = GetX(lChunkFiles,iMaxNumFrams,78)
#print(arFeatData.shape)
if sFeatType == 'A': #Attribute Features
    arAttributeFeatures = np.empty((arFeatData.shape[0],len(lAttribParams)),dtype='float')
    for iParam in range(len(lAttribParams)):
        iAttributeIndx,iNFrams,iNhl,iNhu,sBestParamFile = lAttribParams[iParam]
        Py, yP = Decode(iNhu, iNhl, 2, arFeatData, os.path.join('Temp',sBestParamFile))
        arAttributeFeatures[:,iParam] = Py[:,0]
    arFeat = arAttributeFeatures
elif sFeatType == 'F':
    arFeat = arFeatData[:,312:390]
else:
    arAttributeFeatures = np.empty((arFeatData.shape[0],len(lAttribParams)),dtype='float')
    for iParam in range(len(lAttribParams)):
        iAttributeIndx,iNFrams,iNhl,iNhu,sBestParamFile = lAttribParams[iParam]
        Py, yP = Decode(iNhu, iNhl, 2, arFeatData, os.path.join('Temp',sBestParamFile))
        arAttributeFeatures[:,iParam] = Py[:,0]
    #arFeat = arAttributeFeatures
    arFeat = np.c_[arAttributeFeatures,arFeatData[:,312:390]]
for sPhone in lPhonemes:
    lPhoneTest = []
    vPhoneMask_Train = np.zeros((arFeatData.shape[0]),dtype='bool')
    vPhoneMask_Valid = np.zeros((arFeatData.shape[0]),dtype='bool')
    #vPhoneMask_Test = np.zeros((arFeatData.shape[0]),dtype='bool')
    for iFileIdx, iPSFram, iPEFram in dChunkPhone[sPhone]:
        vPhoneMask_Test = np.zeros((arFeatData.shape[0]),dtype='bool')
        iStartIndx = lStartIndxs[iFileIdx]
        if lChunkTypes[iFileIdx] == 0:
            print(iStartIndx,iPSFram,iPEFram)
            vPhoneMask_Train[iStartIndx+iPSFram: iStartIndx+iPEFram] = True
        elif lChunkTypes[iFileIdx] == 1:
            vPhoneMask_Valid[iStartIndx+iPSFram: iStartIndx+iPEFram] = True
        else:
            vPhoneMask_Test[iStartIndx+iPSFram: iStartIndx+iPEFram] = True
            lPhoneTest.append(arFeat[vPhoneMask_Test])
    print('********************',len(lPhoneTest))
    joblib.dump(arFeat[vPhoneMask_Train],sPhone+'_Train_'+sFeatType+'_'+str(iChunkIndx))
    joblib.dump(arFeat[vPhoneMask_Valid],sPhone+'_Valid_'+sFeatType+'_'+str(iChunkIndx))
    #joblib.dump(arFeat[vPhoneMask_Test],sPhone+'_Test_'+str(iChunkIndx))
    joblib.dump(lPhoneTest,sPhone+'_Test_'+sFeatType+'_'+str(iChunkIndx))
arFeatData,lStartIndxs = GetX(lChunkFiles,iMaxNumFrams,78)
print(arFeatData.shape)

