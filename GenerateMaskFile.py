#!/usr/bin/env python
from __future__ import print_function
from collections import OrderedDict
import sys,os
import gzip,cPickle
import numpy as np
#import hickle as hkl
from random import shuffle
from sklearn.externals import joblib
if len(sys.argv) != 6:
    print('Usage: GenerateMaskFile.py MlfFile ListFiles OutPath PhonemeSet AttributeList')
    sys.exit(1)

fPhones = open('Phonemes.txt','w')
fLogFile = open('GenerateMaskFile.log','w')
sOutPath = sys.argv[3]
#Configures
iSampleRate = 10 #Sampling rate of the feature files in mili seconds
sSavingFormat = 'J'
with open(sys.argv[4]) as fPhonemes:
    lPhonemes = fPhonemes.read().splitlines()

#lPhonemes = ['sil','ao','aa','iy','uw','eh','ih','uh','ah','ax','ae','ey','ay','ow','aw','oy','er','axr','p','b','t','d','k','g','ch','jh','f','v','th','dh','s','z','sh','zh','hh','m','n','em','en','ng','eng','l','el','r','dx','nx','y','w','q']

with open(sys.argv[5]) as fAttributes:
   dAttributes = OrderedDict([tuple([i[0],i[1:]]) for i in [sLine.split() for sLine in fAttributes.read().splitlines()]])
   #dAttributes = dict([tuple(i[0],i[1:]) in sLine.split() [for sLine in fAttributes.read().splitlines()]]) 
lAttributes = dAttributes.keys()
#lAttributes = ['silience','vowels','stops','affricates','fricatives','nasals','liquids','semivowels','voiced','approx','coronal','high','dental','glottal','labial','low','mid','velar','retroflex','anterior','back','continuant','round','tense','Monophthongs','Diphthongs']
#dAttributes = dict(silience=['sil'],\
#                   vowels=['ao','aa','iy','uw','eh','ih','uh','ah','ax','ae','ey','ay','ow','aw','oy','er','axr#'],\
#                   stops=['p','b','t','d','k','g'],\
#                   affricates=['ch','jh'],\
#                   fricatives=['f','v','th','dh','s','z','sh','zh','hh'],\
#                   nasals=['m','n','em','en','ng','eng'],\
#                   liquids=['l','el','r','dx','nx'],\
#                   semivowels=['y','w','q'],\
#                   voiced=['aa','ae','ah','aw','ay','ao','b','d','dh','eh','er','ey','g','ih','iy','jh','l','m','n','ng','ow','oy','r','uh','uw','v','w','y','z'],\
#                   approx=['w','y','l','r'],\
#                   coronal=['d','l','n','s','t','z'],\
#                   high=['ch','ih','iy','jh','sh','uh','uw','y','ow','g','k','ng'],\
#                   dental=['dh','th'],\
#                   glottal=['hh'],\
#                   labial=['b','f','m','p','v','w'],\
#                   low=['aa','ae','aw','ay','oy'],\
#                   mid=['ah','eh','ey','ow'],\
#                   velar=['g','k','ng'],\
#                   retroflex = ['er','r'],\
#                   anterior = ['b','d','dh','f','l','m','n','p','s','t','th','v','z','w'],\
#                   back = ['ay','aa','ah','ao','aw','ow','oy','uh','uw','g','k'],\
#                   continuant = ['ao','aa','iy','ay','r','ey','ih','uw','eh','uh','ah','ae','ow','aw','oy','er','f','v','th','dh','s','z','sh','l','y','w'],\
#                   round = ['aw','ow','uw','ao','uh','v','y','oy','r','w'],\
#                   tense = ['aa','ae','ao','aw','ay','ey','iy','ow','oy','uw','ch','s','sh','f','th','p','t','k','hh'],\
#                   Monophthongs = ['ao','aa','iy','uw','eh','ih','uh','ah','ax','ae'],\
#                   Diphthongs = ['ey','ay','ow','aw','oy'])
enumAttributes = list(enumerate(lAttributes))
#print('phoneme','\t'.join([item for i,item in enumAttributes]),sep='\t',file=fPhones)
iNumAttributes = len(enumAttributes)
dPhonemeMask = {}
dPhonemeCount = {}
for sPhone in lPhonemes:
    arMask = np.zeros(iNumAttributes)
    arTemp = np.zeros(iNumAttributes)
    dPhonemeCount[sPhone] = arTemp
    for i,item in enumAttributes:
        if sPhone in dAttributes[item]:
            arMask[i] = 1
    dPhonemeMask[sPhone] = arMask
iNumPhones = len(lPhonemes)
with open(sys.argv[2]) as fList:
    lListFiles = [os.path.splitext(os.path.basename(sLine))[0] for sLine in fList.read().splitlines()]
with open(sys.argv[1]) as fMlf:
    lMlf = fMlf.read().splitlines()
    lSEInx = [i for i in range(len(lMlf)) if lMlf[i][0]=='"' or lMlf[i][0] == '.'] #Get Start and End index of each file
    for i in range(0,len(lSEInx),2):
        print(int((float(i)/len(lSEInx))*100),'%',end='\r')
        iSinx = lSEInx[i]
        iEinx = lSEInx[i+1]
        sFileName = os.path.splitext(os.path.basename(lMlf[iSinx][1:-1]))[0]
        if sFileName not in lListFiles:
            print('File: ',sFileName,' Not Registered',file=fLogFile)
            continue
        arFileMask = np.asarray([]).reshape(0,iNumAttributes)
        arPhoneFileMask = np.asarray([]).reshape(0,iNumPhones)
        print(sFileName,file=fPhones)
        for j in range(iSinx+1,iEinx):
            lLine = lMlf[j].split()
            iSFram = int(round(float(lLine[0])/iSampleRate**5))
            iEFram = int(round(float(lLine[1])/iSampleRate**5))
            sPhoneme = lLine[2]
            if sPhoneme not in lPhonemes:
                iPhoneIndx = -1
                arPhoneMask = np.zeros(iNumAttributes)
            else:
                iPhoneIndx = lPhonemes.index(sPhoneme)
                arPhoneMask = dPhonemeMask[sPhoneme]
            arPhoneIdxMask = np.zeros((iNumPhones),dtype='int')
            arPhoneIdxMask[iPhoneIndx] = 1
            iNumSamples = iEFram - iSFram
            if iNumSamples == 0:
                print('Duration of '+sPhoneme+' in file ' + sFileName + ' Less than 10msec')
                continue
            #dPhonemeCount[sPhoneme] += arPhoneMask
            arPhoneMask = arPhoneMask.reshape(1,iNumAttributes)
            arPhoneFramMask = arPhoneMask.repeat(iNumSamples,axis=0)
            arFileMask = np.r_[arFileMask,arPhoneFramMask]
            arPhoneIdxMask = arPhoneIdxMask.reshape(1,iNumPhones)
            arPhoneIdxFramMask = arPhoneIdxMask.repeat(iNumSamples,axis=0)
            arPhoneFileMask = np.r_[arPhoneFileMask,arPhoneIdxFramMask]
            print(sPhoneme,arPhoneFramMask.shape,file=fPhones)
            print(arPhoneFramMask[0],file=fPhones)
        joblib.dump(arFileMask,os.path.join(sOutPath,sFileName+'.mask'))
        joblib.dump(arPhoneFileMask,os.path.join(sOutPath,sFileName+'.pmask'))
fPhones.close()
fLogFile.close()
