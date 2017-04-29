from __future__ import print_function
import sys,glob,os
import numpy as np
from sklearn.externals import joblib
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.grid_search import GridSearchCV
import gzip,cPickle

#Config
#sKernel = sys.argv[2]
#sNu = sys.argv[3]
#sGamma = sys.argv[4]
#lPhonemes = ['ao','aa','iy','uw','eh','ih','uh','ah','ae','ey','ay','ow','aw','oy','er','p','b','t','d','k','g','ch','jh','f','v','th','dh','s','z','sh','zh','hh','m','n','ng','l','r','y','w']
if len(sys.argv) != 6:
    print('Usage: TestPhonemeV3.py TargetPhone TrialIndx DataPath phoneList NumAttributes')
    sys.exit(1)

sTargetPhone = sys.argv[1]
sTrial = sys.argv[2]
sDataPath = sys.argv[3]

with open(sys.argv[4]) as fPhonemes:
    lPhonemes = fPhonemes.read().splitlines()
iNumAttributes = int(sys.argv[5])
print(sTargetPhone)
def loadData(sPhone,sDataPath):
    lFiles = glob.glob(os.path.join(sDataPath,sPhone+'_Test_*'))
    #print(lFiles)
    lAllData = []
    for sFile in lFiles:
        if os.path.splitext(sFile)[1] == '':
            #print(sFile)
            lData = joblib.load(sFile)
            #print(len(lData),len(lAllData))
            lAllData += lData
    return lAllData
def loadTrainingData(sPhone,sDataPath,iNumAttributes):
    lFiles = glob.glob(os.path.join(sDataPath,sPhone+'_Train_*'))
    arPhoneData = np.empty((3000000,iNumAttributes),dtype='float')
    iPoint = 0
    for sFile in lFiles:
        if os.path.splitext(sFile)[1] == '':
            #print(sFile)
            arData = joblib.load(sFile)
            iCurSize = arData.shape[0]
            arPhoneData[iPoint:iPoint+iCurSize] = arData
            iPoint +=iCurSize
    return arPhoneData[:iPoint]
with open(sTargetPhone+'_'+sTrial+'.log') as fPhoneLog:
    Line = fPhoneLog.read().splitlines()[-1]
    sLine = Line.split()
    sKernel = sLine[5]
    sNu = sLine [3]
    sGamma = sLine[4]
    if Line.find(',') == -1:
        indx = [int(i.replace('[','').replace(',','').replace(']','')) for i in sLine[7:]]
    else:
        indx = [int(i.replace('[','').replace(',','').replace(']','')) for i in sLine[6:]]
print('Training Model')
arTraining = loadTrainingData(sTargetPhone,sDataPath,iNumAttributes)#joblib.load(sTargetPhone+'.jbl')
scal = StandardScaler()
arTraining_std = scal.fit_transform(arTraining)[:,indx]
#clsf = OneClassSVM(nu = float(sNu), gamma = float(sGamma), kernel=sKernel)
#clsf.fit(arTraining_std)
sModel = sTargetPhone+'_'+sTrial+'.model'
with open(sModel) as fModel:
    clsf = cPickle.load(fModel)
print('Testing Model')
iAllHits = 0.0
iAllMiss = 0.0
fRes = open(sTargetPhone+'_'+sTrial+'.res','w')
inClassMiss = 0.0
inClassHit = 0.0
OutClassMiss = 0.0
OutClassHit = 0.0
for sPhone in lPhonemes:
    #with gzip.open(sPhone+'_test.pkl') as fPhone:
        #lData = cPickle.load(fPhone)
    lData = loadData(sPhone,sDataPath)
    print(sPhone)
    iHits = 0.0
    iMiss = 0.0
    for sPhoneData in lData:
        if sPhoneData.shape[0] < 4:
            continue
        sPhoneData_std = scal.transform(sPhoneData)[:,indx]
        y_predict = clsf.predict(sPhoneData_std)
        iNumInClassFrams = np.where(y_predict == 1)[0].shape[0]
        iNumOutClassFrams = np.where(y_predict == -1)[0].shape[0]
        if sPhone == sTargetPhone:
            if iNumInClassFrams >= iNumOutClassFrams:
                iHits += 1
                inClassHit += 1
            else:
                iMiss += 1
                inClassMiss += 1
        else:
            if iNumInClassFrams < iNumOutClassFrams:
                iHits += 1
                OutClassHit += 1
            else:
                iMiss += 1
                OutClassMiss += 1
    iAllHits += iHits
    iAllMiss += iMiss
    print(sTargetPhone,sPhone,iHits,iMiss,iHits/(iHits+iMiss),file=fRes)
print ('#############################################################',file=fRes)
print (sTargetPhone,iAllHits/(iAllHits+iAllMiss),inClassHit/(inClassHit+inClassMiss),OutClassHit/(OutClassHit+OutClassMiss),file=fRes)
fRes.close()
