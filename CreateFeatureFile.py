#!/usr/bin/env python

from __future__ import print_function
import sys, gzip, cPickle
import numpy as np
import scipy.io.wavfile as wav
#import hickle as hkl
from sklearn.externals import joblib
from python_speech_features import mfcc,fbank,logfbank,ssc,delta

#Configs
iDeltaWindow = 2
iAccWindow = 2
iThirdWindow = 2

if len(sys.argv) != 6:
    print('Usage: CreateFeatureFile listOfFiles Type(mfcc,fbank,lfbank,ssc) Drivatives(N,D,A,T) StandrizeIt(1,0) iSize')
    sys.exit(1)

sFList = sys.argv[1]
sFeatureType = sys.argv[2]
sDrivatives = sys.argv[3]
bStandrize = bool(int(sys.argv[4]))
iSize = int(sys.argv[5])

with open(sFList,'r') as fList:
    lWavFiles = fList.read().splitlines()
    for sLine in lWavFiles:
        sWavFile,sFeatureFile = sLine.split()
        print(sWavFile)
        iRate,lSamples = wav.read(sWavFile)
        print(sWavFile,end = '\r')
        #Ceating Features
        if sFeatureType == 'mfcc':
            aFeatures = mfcc(lSamples,iRate)
        elif sFeatureType == 'fbank':
            aFeatures = fbank(lSamples,iRate)
        elif sFeatureType == 'lfbank':
            aFeatures = logfbank(lSamples,iRate,nfilt=iSize)
        elif sFeatureType == 'ssc':
            aFeatures = ssc(lSamples,iRate)
        else:
            print('Error: Unknown Feature Type sFeatureType')
            sys.exit(1)
        
        #Computing Time Drivatives
        if sDrivatives == 'D':
            aDFeatures = delta(aFeatures,iDeltaWindow)
            aFeatures = np.c_[aFeatures,aDFeatures]
        elif sDrivatives == 'A':
            aDFeatures = delta(aFeatures,iDeltaWindow)
            aAFeatures = delta(aDFeatures,iAccWindow)
            aFeatures = np.c_[aFeatures,aDFeatures,aAFeatures]
        elif sDrivatives == 'T':
            aDFeatures = delta(aFeatures,iDeltaWindow)
            aAFeatures = delta(aDFeatures,iAccWindow)
            aTFeatures = delta(aAFeatures,iThirdWindow)
            aFeatures = np.c_[aFeatures,aDFeatures,aAFeatures,aTFeatures]
        
        #Standrizing Data
        if bStandrize:
            vMean = np.mean(aFeatures,0)
            vStd  = np.std(aFeatures,0)
            aFeatures = (aFeatures-vMean)/vStd

        #Save Feature File
        #with gzip.open(sFeatureFile,'w') as fFeatureFile:
        #    cPickle.dump(aFeatures,fFeatureFile)
        #hkl.dump(aFeatures, sFeatureFile, mode='w', compression='gzip')
        joblib.dump(aFeatures,sFeatureFile,3)
            #print('hello')
        #print(aFeatures.shape,np.amax(aFeatures),np.amin(aFeatures),aFeatures[1:3,:])
        
