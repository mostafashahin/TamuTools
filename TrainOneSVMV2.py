from __future__ import print_function
import sys, os
from copy import deepcopy
import numpy as np
import cPickle
import glob
from sklearn.externals import joblib
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score
from sklearn.grid_search import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.base import clone

def scorer(etimator,X,Y):
    y_pre = etimator.predict(X)
    return accuracy_score(Y,y_pre)
def loadData(sPhone,sType,sDataPath,iNumAttributes):
    lFiles = glob.glob(os.path.join(sDataPath,sPhone+'_'+sType+'_*'))
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
#lPhonemes = ['ao','aa','iy','uw','eh','ih','uh','ah','ae','ey','ay','ow','aw','oy','er','p','b','t','d','k','g','ch','jh','f','v','th','dh','s','z','sh','zh','hh','m','n','ng','l','r','y','w']
iNumValidationSamples = 2000
if len(sys.argv) != 6:
    print('Usage: TrainOneSVMV2.py TargetPhone TrailIndx PhoneAttribPath phoneList NumAttributes')
    sys.exit(1)

sTargetPhone = sys.argv[1]
sTrial = sys.argv[2]
sDataPath = sys.argv[3]
fLog = open(sTargetPhone+'_'+sTrial+'.log','w')
with open(sys.argv[4]) as fPhonemes:
    lPhonemes = fPhonemes.read().splitlines()
iNumAttributes = int(sys.argv[5])
#arTraining = joblib.load(sTargetPhone+'.jbl')
arTraining = loadData(sTargetPhone,'Train',sDataPath,iNumAttributes)
iNumFeatures = arTraining.shape[1]
arValidation = np.zeros((iNumValidationSamples,iNumFeatures))
vValidPhonemes = np.zeros((iNumValidationSamples),dtype='str')
iNumInClass = int(0.3*iNumValidationSamples)
iNumOutClass = iNumValidationSamples - iNumInClass
#arValidInClass = joblib.load(sTargetPhone+'_valid.jbl')
arValidInClass = loadData(sTargetPhone,'Valid',sDataPath,iNumAttributes)
iNumExist = arValidInClass.shape[0]
iUsedInClass = min(iNumInClass,iNumExist)
np.random.shuffle(arValidInClass)
arValidation[:iUsedInClass,:] = arValidInClass[:iUsedInClass,:]
vValidPhonemes[:iUsedInClass] = sTargetPhone
i = iUsedInClass

iNumPerPhone = int(iNumOutClass/len(lPhonemes)-1)
print(sTargetPhone)
for sPhone in lPhonemes:
    if sPhone == sTargetPhone:
        continue
    #print(sPhone,iNumPerPhone)
    #arValidOutClass = joblib.load(sPhone+'_valid.jbl')
    arValidOutClass = loadData(sPhone,'Valid',sDataPath,iNumAttributes)
    iNumExist=arValidOutClass.shape[0]
    iUsed = min(iNumPerPhone,iNumExist)
    np.random.shuffle(arValidOutClass)
    arValidation[i:i+iUsed,:] = arValidOutClass[:iUsed,:]
    vValidPhonemes[i:i+iUsed] = sPhone
    i = i + iUsed
#if i < iNumValidationSamples:
#    iDiff = iNumValidationSamples - i
#    arValidation[i:i+iDiff,:] = arValidOutClass[iNumPerPhone:iNumPerPhone+iDiff,:]
#    vValidPhonemes[i:i+iDiff] = sPhone
arValidation = arValidation[:i,:] 
y_valid_ref = np.r_[np.zeros(iUsedInClass)+1,np.zeros(i-iUsedInClass)-1]
pca = PCA(n_components=8)

#Standarizing Data
scal = StandardScaler()
arTraining_std = scal.fit_transform(arTraining)
arValidation_std = scal.transform(arValidation)
pca.fit(arTraining_std)
arTraining_pca = pca.transform(arTraining_std)
arValidation_pca = pca.transform(arValidation_std)

arCVData = np.r_[arTraining_std,arValidation_std]
arCVLab = np.r_[np.zeros(arTraining.shape[0])+1,y_valid_ref]
tCVIndxs = (range(0,arTraining_std.shape[0]),range(arTraining_std.shape[0],arTraining_std.shape[0]+arValidation_std.shape[0]))
dParams = {'nu':[0.1,0.15,0.2]}
clsf = OneClassSVM(coef0=0.0,verbose=False)
clsfRFE = OneClassSVM(kernel='linear',nu=0.00001)
selector = RFE(clsfRFE)
selector.fit(arTraining_std,np.ones(arTraining.shape[0]))
with open(sTargetPhone+'_'+sTrial+'.rank','w') as fRank:
    Indx = np.where(selector.ranking_==1)[0]
    print(Indx,file=fRank)
fMaxAcc = 0.0
sMAXparam=''
clfMax = ''
idxmax = 0
for indx in (range(arTraining_std.shape[1]),Indx):
    for kernel in ['rbf','sigmoid']:#['poly', 'rbf', 'sigmoid']:
        for nu in [0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.1]:#,0.15,0.2,0.25,0.3,0.35,0.4]:
            for gam in [0.000001,0.00001,0.0001,0.001,0.01,0.1]:#,0.3,0.5]:#,1,10,100]:
                #print ('nu=',nu,'gamma=',gam,'kernel=',kernel)
                param = str(nu)+' '+str(gam) + ' '+ kernel
                clsf.set_params(**{'nu':nu,'gamma':gam,'kernel':kernel})
        #gsC    lsf = GridSearchCV(clsf,dParams,cv=tCVIndxs,scoring='scorer')
                #print('Training The Model')
                arTr = arTraining_std[:,indx]
                clsf.fit(arTr)
        #cls    f.fit(arCVData,arCVLab)
                #print('Prediction')
                arV = arValidation_std[:,indx]
                y_pre = clsf.predict(arV)
                param = param + ' ' + str(indx).replace('\n','')
                #print (accuracy_score(y_valid_ref,y_pre))
                fCurAcc = f1_score(y_valid_ref,y_pre)#accuracy_score(y_valid_ref,y_pre)
                fscore1 = accuracy_score(y_valid_ref[:iNumInClass],y_pre[:iNumInClass])
                fscore2 = accuracy_score(y_valid_ref[iNumInClass:],y_pre[iNumInClass:])
                fLog.write(param+'\t'+str(fCurAcc)+'\t'+str(fscore1)+'\t'+str(fscore2)+'\n')
                fLog.flush()
                print(param,fCurAcc)
                if fCurAcc > fMaxAcc:
                    fMaxAcc = fCurAcc
                    sMAXparam = param
                    clfMax = deepcopy(clsf)
                    #y_pre = clfMax.predict(arValidation_std[:,indx])
                    print(clfMax)
                    idxmax = indx
                    print (len(indx))
print ('BestParam = ',fMaxAcc,sMAXparam,file=fLog)
print ('BestParam = ',fMaxAcc,sMAXparam)
fModel = open(sTargetPhone+'_'+sTrial+'.model','w')
cPickle.dump(clfMax,fModel)
fModel.close()
fModel = open(sTargetPhone+'_'+sTrial+'.model','r')
clsf2 = cPickle.load(fModel)
print(len(idxmax))
y_pre = clsf2.predict(arValidation_std[:,idxmax])
fCurAcc = f1_score(y_valid_ref,y_pre)
print (fCurAcc)
fLog.close()
                
            #print(confusion_matrix(y_valid_ref,y_pre))
