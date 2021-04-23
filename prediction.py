import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import svm
from sklearn import model_selection
import matplotlib.pyplot as plt
from sklearn import metrics
%matplotlib inline

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import Lasso
import pickle
from sklearn import ensemble
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor

#SVD Reconstruction: Training Data
utrain = pd.read_csv('utrain.dat', sep = ",",header =None)
vtrain = pd.read_csv('vtrain.dat', sep = ",",header =None)
sgmatrain = pd.read_csv('sgmatrain.dat', sep = ",",header =None)

utest = pd.read_csv('utest.dat', sep = ",",header =None)
vtest = pd.read_csv('vtest.dat', sep = ",",header =None)
sgmatest = pd.read_csv('sgmatest.dat', sep = ",",header =None)

features = pd.read_csv('features23.dat', sep = ",",header =None)
weights = pd.read_csv('weights.dat', sep = ",",header =None)
ticktrain = np.dot(np.dot(utrain,sgmatrain),vtrain.T)
ticktest = np.dot(np.dot(utest,sgmatest),vtest.T)

reconst = np.concatenate((features, np.r_[ticktrain,ticktest],weights),axis=1)
colnames = 'Feature_1	Feature_2	Feature_3	Feature_4	Feature_5	Feature_6	Feature_7	Feature_8	Feature_9	Feature_10	Feature_11	Feature_12	Feature_13	Feature_14	Feature_15	Feature_16	Feature_17	Feature_18	Feature_19	Feature_20	Feature_21	Feature_22	Feature_23	Feature_24	Feature_25	Ret_MinusTwo	Ret_MinusOne	Ret_2	Ret_3	Ret_4	Ret_5	Ret_6	Ret_7	Ret_8	Ret_9	Ret_10	Ret_11	Ret_12	Ret_13	Ret_14	Ret_15	Ret_16	Ret_17	Ret_18	Ret_19	Ret_20	Ret_21	Ret_22	Ret_23	Ret_24	Ret_25	Ret_26	Ret_27	Ret_28	Ret_29	Ret_30	Ret_31	Ret_32	Ret_33	Ret_34	Ret_35	Ret_36	Ret_37	Ret_38	Ret_39	Ret_40	Ret_41	Ret_42	Ret_43	Ret_44	Ret_45	Ret_46	Ret_47	Ret_48	Ret_49	Ret_50	Ret_51	Ret_52	Ret_53	Ret_54	Ret_55	Ret_56	Ret_57	Ret_58	Ret_59	Ret_60	Ret_61	Ret_62	Ret_63	Ret_64	Ret_65	Ret_66	Ret_67	Ret_68	Ret_69	Ret_70	Ret_71	Ret_72	Ret_73	Ret_74	Ret_75	Ret_76	Ret_77	Ret_78	Ret_79	Ret_80	Ret_81	Ret_82	Ret_83	Ret_84	Ret_85	Ret_86	Ret_87	Ret_88	Ret_89	Ret_90	Ret_91	Ret_92	Ret_93	Ret_94	Ret_95	Ret_96	Ret_97	Ret_98	Ret_99	Ret_100	Ret_101	Ret_102	Ret_103	Ret_104	Ret_105	Ret_106	Ret_107	Ret_108	Ret_109	Ret_110	Ret_111	Ret_112	Ret_113	Ret_114	Ret_115	Ret_116	Ret_117	Ret_118	Ret_119	Ret_120	Ret_121	Ret_122	Ret_123	Ret_124	Ret_125	Ret_126	Ret_127	Ret_128	Ret_129	Ret_130	Ret_131	Ret_132	Ret_133	Ret_134	Ret_135	Ret_136	Ret_137	Ret_138	Ret_139	Ret_140	Ret_141	Ret_142	Ret_143	Ret_144	Ret_145	Ret_146	Ret_147	Ret_148	Ret_149	Ret_150	Ret_151	Ret_152	Ret_153	Ret_154	Ret_155	Ret_156	Ret_157	Ret_158	Ret_159	Ret_160	Ret_161	Ret_162	Ret_163	Ret_164	Ret_165	Ret_166	Ret_167	Ret_168	Ret_169	Ret_170	Ret_171	Ret_172	Ret_173	Ret_174	Ret_175	Ret_176	Ret_177	Ret_178	Ret_179	Ret_180	Ret_PlusOne	Ret_PlusTwo	Weight_Intraday	Weight_Daily'
colnames = colnames.split()
del(colnames[0])
del(colnames[14])
df = pd.DataFrame(reconst,columns=colnames)

# split train and test
dttrain = reconst[:30000,:]
dttest = reconst[30000:40000,:]
print dttrain.shape
print dttest.shape

intradytr = dttrain[:,25:]
intradyts = dttest[:,25:]
print intradytr.shape
print intradyts.shape

yweights = dttrain[:,-4:]
intradytr = np.delete(intradytr,[179,180,181,182],1)
intradyts = np.delete(intradyts,[179,180,181,182],1)
Xtrtrue = intradytr
Xtrtemp = intradytr[:,0:119]
Xtstrue = intradyts
Xtstemp = intradyts[:,0:119]

#Data Preprocessing: Scaling
# normalize the dataset
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
#Xscale = scaler.fit_transform(Xtrue)

#Training
from sklearn import neural_network
from sklearn import linear_model
from sklearn import tree
#model1 = ExtraTreesRegressor(n_estimators=10, max_features=32,random_state=0)
#model2 = KNeighborsRegressor()
#model3 = LinearRegression()
#model = MultiOutputRegressor(linear_model.SGDRegressor(random_state=0,loss='squared_loss',fit_intercept=False))
#model = neural_network.MLPRegressor(hidden_layer_sizes = 500, random_state=0,learning_rate='invscaling')
model1 = LinearRegression()
model2 = tree.DecisionTreeRegressor()
model3 = MultiOutputRegressor(linear_model.SGDRegressor(random_state=0,loss='squared_loss',fit_intercept=False))
modelcol1 = []
modelcol2 = []
modelcol3 = []
scalercolX = []
scalercolY = []
for i in range(1,7):
    windowX = range(0,109+i*10)
    windowY = range(109+i*10, 109+i*10+10)
    Xtrain = Xtrtrue[:,windowX]
    Ytrain = Xtrtrue[:,windowY]
    scaler.fit(Xtrain)
    Xtrain = scaler.transform(Xtrain)
    scalercolX.append(pickle.dumps(scaler))
    scaler.fit(Ytrain)
    Ytrain = scaler.transform(Ytrain)
    scalercolY.append(pickle.dumps(scaler))
    model1.fit(Xtrain, Ytrain)
    modelcol1.append(pickle.dumps(model1))
    model2.fit(Xtrain, Ytrain)
    modelcol2.append(pickle.dumps(model2))
    model3.fit(Xtrain, Ytrain)
    modelcol3.append(pickle.dumps(model3))
    #model2.fit(Xtrain, Ytrain)
    #model3.fit(Xtrain, Ytrain)
    #windowXpred = range(10, 109+i*10+10)
    #Xpred = X[:,windowXpred]
    #Ypred1 = model1.predict(Xpred)
    #Ypred2 = model2.predict(Xpred)
    #Ypred3 = model3.predict(Xpred)
    #X = np.c_[X,Ypred2]
    
    
    
#Testing
Xtest = Xtstemp
for i in range(0,6):
    #windowX = range(0,109+(i+1)*10)
    #Xtest = Xtstemp[:,windowX]
    cscalerX = pickle.loads(scalercolX[i])
    Xpred = cscalerX.transform(Xtest)
    model1 = pickle.loads(modelcol1[i])
    model2 = pickle.loads(modelcol2[i])
    model3 = pickle.loads(modelcol3[i])
    Ypred1 = model1.predict(Xpred)
    Ypred2 = model2.predict(Xpred)
    Ypred3 = model3.predict(Xpred)
    #model2.fit(Xtrain, Ytrain)
    #model3.fit(Xtrain, Ytrain)
    #windowXpred = range(10, 109+i*10+10)
    #Xpred = X[:,windowXpred]
    #Ypred1 = model1.predict(Xpred)
    #Ypred2 = model2.predict(Xpred)
    #Ypred3 = model3.predict(Xpred)
    
    Ypred = Ypred1*9/10 + Ypred2/20 + Ypred3/20
    cscalerY = pickle.loads(scalercolY[i])
    Yinv = cscalerY.inverse_transform(Ypred)
    Xtest = np.c_[Xtest,Yinv]
    
samplenum = np.random.choice(10,10)
print samplenum  

xr = range(119,179)
f, ax = plt.subplots(figsize=(9, 8))
plt.plot(xr,Xtest[samplenum,119:].T)
plt.xlabel('t',fontsize=16)
plt.ylabel('Returns',fontsize=16)
plt.title('Prediction: Return of t_121 to t_180',fontsize=16)
yticks = np.linspace(-0.005, 0.005, 5)
ax.set_yticks(yticks)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.savefig('sample1')

f, ax = plt.subplots(figsize=(9, 8))
plt.plot(xr,Xtstrue[samplenum,119:].T)
plt.xlabel('t',fontsize=16)
plt.ylabel('Returns',fontsize=16)
plt.title('True Values: Return of t_121 to t_180',fontsize=16)
yticks = np.linspace(-0.005, 0.005, 5)
ax.set_yticks(yticks)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.savefig('sample2')


# PCA to Select Necessary Features
featuretr = dttrain[:,:23]
featurets = dttest[:,:23]
from sklearn.decomposition import PCA
from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(featuretr)
feascale = scaler.transform(featuretr)
numcol = featuretr.shape[1]
print numcol
colidx = range(numcol)
print colidx
colidx = list(np.array(colidx) + 1)
print colidx
for i in colidx:
    pca = PCA(n_components=i)
    pca_result = pca.fit_transform(feascale)
    print 'Num of Components %2.0f' % i, 'Variance', sum(pca.explained_variance_ratio_)* 100


pca = PCA(n_components=18)
pca_result = pca.fit_transform(feascale)
dfpca = pd.DataFrame(pca.components_,columns=df.columns[:23], index = ['PC-1','PC-2','PC-3','PC-4','PC-5','PC-6','PC-7',\
                                                                    'PC-8','PC-9','PC-10','PC-11','PC-12','PC-13','PC-14',\
                                                                    'PC-15','PC-16','PC-17','PC-18'])


pcafea = dfpca.abs().idxmax(axis=1)
print pcafea
print pcafea.shape
feanew = df[pcafea].values
feanewtr = feanew[:30000,:]
feanewts = feanew[30000:40000,:]
feanew.shape

returnD_minor_21 = np.c_[df['Ret_MinusTwo'].values,df['Ret_MinusOne'].values]
returnD_plus_12 = np.c_[df['Ret_PlusOne'].values,df['Ret_PlusTwo'].values]

return_weight_plus_12=df["Weight_Daily"].iloc[30000:40000].values
return_weight_intraday=df["Weight_Intraday"].iloc[30000:40000].values

returnDtr_minor_21 = returnD_minor_21[:30000,:]
returnDts_minor_21 = returnD_minor_21[30000:40000,:]
returnDtr_plus_12 = returnD_plus_12[:30000,:]
returnDts_plus_12 = returnD_plus_12[30000:40000,:]

trainXstep2 = np.c_[feanewtr,returnDtr_minor_21,Xtrtrue]
trainYstep2 = returnDtr_plus_12
testYstep2 = returnDts_plus_12

colXstep2 = []
colYstep2 = []
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
#Xscale = scaler.fit_transform(Xtrue)
scaler.fit(trainXstep2)
trainXstep2 = scaler.transform(trainXstep2)
colXstep2 = pickle.dumps(scaler)
scaler.fit(trainYstep2)
trainYstep2 = scaler.transform(trainYstep2)
colYstep2 = pickle.dumps(scaler)

# predicting
#testY = model.predict(testXstep2)
testY1 = model1.predict(testXstep2)
testY2 = model2.predict(testXstep2)
testY3 = model3.predict(testXstep2)
testY = testY1*9/10 + testY2*1/20 + testY3*1/20
scalerYstep2 = pickle.loads(colYstep2)


# Training
model1 = LinearRegression()
model2 = tree.DecisionTreeRegressor()
model3 = MultiOutputRegressor(linear_model.SGDRegressor(random_state=0,loss='squared_loss',fit_intercept=False))
model1.fit(trainXstep2,trainYstep2)
model2.fit(trainXstep2,trainYstep2)
model3.fit(trainXstep2,trainYstep2)
testXstep2 = np.c_[feanewts,returnDts_minor_21,Xtest]
scalerXstep2 = pickle.loads(colXstep2)
testXstep2 = scalerXstep2.transform(testXstep2)


testYstep2 = scalerYstep2.inverse_transform(testY)
print np.linalg.norm(testYstep2-returnDts_plus_12,'fro')/np.linalg.norm(returnDts_plus_12,'fro')
print np.linalg.norm(testYstep2,'fro')


Dplus12 = pd.read_csv('Dplus12.dat', sep = ",",header =None)

trueDts_plus_12 = Dplus12.values
test_plus_12 = trueDts_plus_12[30000:40000,:]

xnum = range(1,10001)
f, ax = plt.subplots(figsize=(9, 8))
o1 = plt.scatter(xnum,testYstep2[:,0],color='blue')
o2 = plt.scatter(xnum,test_plus_12[:,0],color='red')
plt.xlabel('Numbers',fontsize=16)
plt.ylabel('Returns',fontsize=16)
plt.title('Return of +D1',fontsize=16)
yticks = np.linspace(-0.5, 0.5, 5)
ax.set_yticks(yticks)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.savefig('+D1')

xnum = range(1,10001)
f, ax = plt.subplots(figsize=(9, 8))
o1 = plt.scatter(xnum,testYstep2[:,1],color='blue')
o2 = plt.scatter(xnum,test_plus_12[:,1],color='red')
plt.xlabel('Numbers',fontsize=16)
plt.ylabel('Returns',fontsize=16)
plt.title('Return of +D2',fontsize=16)
yticks = np.linspace(-0.5, 0.5, 5)
ax.set_yticks(yticks)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.savefig('+D2')

n=10000
wmae=0
for i in range(0,10000):
     wmae=wmae+abs(testYstep2[i,1]-test_plus_12[i,1])*return_weight_plus_12[i]
wmae_n=wmae/n
print wmae_n

n=10000
wmae=0
for i in range(0,10000):
     wmae=wmae+abs(testYstep2[i,0]-test_plus_12[i,0])*return_weight_plus_12[i]
wmae_n=wmae/n
print wmae_n

from statsmodels.tsa.arima_model import ARMA
Xtrain_ARMA = pd.DataFrame(Xtrtrue.T)
Xtrain_ARMA_1=Xtrain_ARMA.iloc[0:119,:]
Xtrain_ARMA_2=Xtrain_ARMA.iloc[119:179,:]
import statsmodels.api as sm

model4 = sm.tsa
Ytrain=Xtrain_ARMA_2.iloc[:,0]
Xtrain = Xtrain_ARMA_1.iloc[:,0]
model4=model4.ARMA(Xtrain,(10,5)).fit()

from pandas.tseries.offsets import *
# out-sample predict
start_date = Xtrain.index[-1] + Day(1)
end_date = Xtrain.index[-1] + Day(60)
y_forecast = model4.predict(start_date.isoformat(), end_date.isoformat())
print(y_forecast)

# plot
f, ax = plt.subplots(figsize=(9, 8))
index=np.arange(120,180,1)
plt.scatter(index,Ytrain)
plt.scatter(index,y_forecast)
plt.ylim((-0.002,0.002))
plt.title("ARMA model for stock 1",fontsize=16)
plt.xlabel("t",fontsize=16)
plt.ylabel("Returns",fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.savefig('+D2')

# Time Calculation for stock 3

tic = time.clock()

y_forecast={}
model4 = sm.tsa
Ytrain=Xtrain_ARMA_2.iloc[:,2]
Xtrain = Xtrain_ARMA_1.iloc[:,2]
model4=model4.ARMA(Xtrain,(20,4)).fit()

start_date = Xtrain.index[-1] + Day(1)
end_date = Xtrain.index[-1] + Day(60)
y_forecast[2] = model4.predict(start_date.isoformat(), end_date.isoformat())

toc = time.clock()
runtime= toc - tic
print "runtime",runtime

f, ax = plt.subplots(figsize=(9, 8))
index=np.arange(120,180,1)
plt.scatter(index,Ytrain)
plt.scatter(index,y_forecast[2])
plt.ylim((-0.002,0.002))
plt.title("ARMA model for stock 2",fontsize=16)
plt.xlabel("t",fontsize=16)
plt.ylabel("Returns",fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.savefig('stock2')

                                                  
                                                  
                                                  
                                                  



