import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from pylab import *
import shap
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor
import lightgbm as lgb
from sklearn.metrics import r2_score,explained_variance_score,mean_absolute_error
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import Pipeline
from catboost import CatBoostRegressor

font1 = {'family': 'Times New Roman','weight': 'normal','size': 13,}
font2 = {'family': 'STSong','weight': 'normal','size': 13,}
fontsize1=13

# 设置字体，以作图显示中文
mpl.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus']=False
# 设置显示属性
pd.set_option('display.max_columns',1000)
pd.set_option('display.max_rows',100)

np.set_printoptions(suppress=True)
np.set_printoptions(suppress=True)
pd.set_option('precision',4)
np.set_printoptions(precision=4)
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)
pd.set_option('display.width',2000)           #宽度



def MSE(y_pre_train,Y_train):
    return sum((y_pre_train-Y_train)**2)/len(Y_train)

# def R2(y_pre,y_real):
#     # return sum((y_pre - y_real.mean()) ** 2) / sum((y_real - y_real.mean()) ** 2)
#     # print(y_real.mean())
#     return 1-(sum((y_real-y_pre)**2)/sum((y_real-y_real.mean())**2))

def MAE(y_pre,y_real):
    return sum(abs(y_real-y_pre))/len(y_real)       #abs为绝对值

def PED(y_pre,y_real,detal = 0.15):                 #PED是误差分布直方图
    error = abs(y_real - y_pre)
    num = 0
    for i in error:
        if i <= detal:
            num = num+1
    ped = num / len(error) * 100

    return ped


# 导入数据'甲苯','汽油重芳烃',
filename= r'C:\Users\zmj\Desktop\数据1.xlsx'
names = ['辛烷值', '抗爆值','硫含量', '苯含量', '烯烃', '芳烃','加氢汽油', 'MTBE', '车用异辛烷', '汽油重芳烃', '乙苯', '甲苯', '二甲苯']
#names = [ '研究法辛烷值', '抗爆性','10%蒸发温度','50%蒸发温度','90%蒸发温度','终馏点','蒸气压','硫含量', '苯含量', '烯烃含量', '芳烃含量','氧含量', '密度','导电性','加氢汽油', 'MTBE', '车用异辛烷', '汽油重芳烃', '乙苯', '甲苯'], '氧含量','密度', '导电性''终馏点', '蒸气压',
data = pd.read_excel(filename)#'汽油重芳烃','甲苯',
index = ['辛烷值', '抗爆值','硫含量', '苯含量', '烯烃', '芳烃','加氢汽油', 'MTBE', '车用异辛烷', '汽油重芳烃', '乙苯', '甲苯', '二甲苯']


#data = pd.read_excel(filename , names=names)
#print('data.shape:{}'.format(data.shape)) '乙苯','汽油重芳烃',
#filename = '../dataset/11.xls'
#data = pd.read_excel(filename)#,header=None).iloc[:,2:]
#index = []
#data = data[index]
#print(data)
#a = np.log10(data['加氢汽油'])
#print(min(a))
#data['车用异辛烷'] = np.log10(data['车用异辛烷'])


# 分离数据集
X=data.iloc[:,6:].values
Y=data.iloc[:,:6].values

n_train = int(len(data) * 0.8)
X_train = X[:n_train]
Y_train = Y[:n_train]
print('X_train.shape:{}'.format(X_train.shape))


X_test = X[n_train:]
Y_test = Y[n_train:]



#validation_size = 0.03
# seed = 0

# 评估算法 - 评估标准
num_folds = 10
scoring = 'neg_mean_squared_error'
#scoring = 'neg_mean_absolute_error'
#scoring ='r2'

# 算法对比

pipelines = {}
#pipelines['MLR'] = Pipeline([('Scaler', StandardScaler()), ('MLR', MultiOutputRegressor(LinearRegression()))])
#pipelines['RR'] = Pipeline([('Scaler', StandardScaler()), ('RR', MultiOutputRegressor(Ridge(alpha=1)))])
#pipelines['Lasso'] = Pipeline([('Scaler', StandardScaler()), ('LASSO', MultiOutputRegressor(Lasso(alpha=0.01)))])
#pipelines['EN'] = Pipeline([('Scaler', StandardScaler()), ('EN', MultiOutputRegressor(ElasticNet(alpha=0.01, l1_ratio=0.5)))])weights='distance',algorithm='kd_tree',leaf_size=90, metric='minkowski', p = 1,n_jobs=1
#pipelines['KNN'] = Pipeline([('Scaler', StandardScaler()), ('KNN', MultiOutputRegressor(KNeighborsRegressor(n_neighbors=2,weights='distance',p=1)))])
#pipelines['CART'] = Pipeline([('CART', MultiOutputRegressor(DecisionTreeRegressor(max_leaf_nodes=40,min_samples_leaf=10)))])
#pipelines['RF'] = Pipeline([('RF', MultiOutputRegressor(RandomForestRegressor(n_estimators=110,min_samples_split=4,max_depth=850)))])
#pipelines['SVM'] = Pipeline([('Scaler', StandardScaler()), ('SVM', MultiOutputRegressor(SVR(kernel='rbf',C=20, max_iter=-1)))])

pipelines['Lightgbm'] = MultiOutputRegressor(LGBMRegressor(
    #boosting_type='gbdt',
    #num_leaves=52,
    max_depth=9,
    learning_rate=0.08050,
    n_estimators=795,
    #objective='regression', # 默认是二分类
    min_data_in_leaf=6,
    min_child_weight=5,
    # subsample=1.0,
    # subsample_freq=1,
    # colsample_bytree=1.0,
    reg_alpha=0.00257,
    reg_lambda=0.00364,
    #random_state=None,
    #silent=True
))

#pipelines['CatBoost'] = MultiOutputRegressor( CatBoostRegressor(
    #iterations=858,
    #learning_rate=0.091,
    #depth=9
    #l2_leaf_reg=2.71
    #subsample=0.9
    #one_hot_max_size=11
 #))


num_algo = len(pipelines)
mse = [[],[]]
Fh = []
mae2 = [[],[]]
PED1 = [[],[]]
R2 = [[],[]]
num_label = []

for num in range(1):
    num = '第'+str(num)+'次'
    num_label.append(num)
    #X_train, X_validation, Y_train1, Y_validation1 = train_test_split(X, Y, test_size=validation_size)
    #Y_train = Y_train1.copy()
    #Y_validation = Y_validation1.copy()


    #print(Y_train)
    #Y_train[:, 1] = 10 ** (Y_train[:, 1])
    #Y_validation[:, 1] = 10 ** (Y_validation1[:, 1])

    key_label = []
    MSE_train = np.zeros((num_algo ,6))
    MSE_test = np.zeros((num_algo , 6))
    MAE_train = np.zeros((num_algo ,6))
    MAE_test = np.zeros((num_algo , 6))
    R2_train = np.zeros((num_algo , 6))
    R2_test = np.zeros((num_algo , 6))
    PED_train = np.zeros((num_algo ,6))
    PED_test = np.zeros((num_algo , 6))
    j = 0
    for key in pipelines:
        key_label.append(key)
        if key == 'Lightgbm':
            model = pipelines[key].fit(X_train,Y_train)  # , eval_set=[(X_train, Y_train), (X_validation,Y_validation)], verbose=100, early_stopping_rounds=50)
            #model = pipelines[key].fit(X_train,Y_train1)

        else:
            model = pipelines[key].fit(X_train, Y_train)
            #model = pipelines[key].fit(X_train, Y_train1)
        y_pre_train = model.predict(X_train)
        y_pre_test = model.predict(X_test)



            #model = pipelines[key].fit(X_train,Y_train1)#, eval_set=[(X_train, Y_train), (X_validation,Y_validation)], verbose=100, early_stopping_rounds=50)

        #else:
            #model = pipelines[key].fit(X_train,Y_train1)
        #y_pre_train = model.predict(X_train)
        #y_pre_test = model.predict(X_validation)

        #y_pre_train[:, 1] = 10 ** (y_pre_train[:, 1])

        #y_pre_test[:, 1] = 10 ** (y_pre_test[:, 1])


        for i in range(len(y_pre_train[0])):
            print(key,i)
            error_ensemble = (y_pre_test[:, i] -  Y_test[:, i])
            #error_ensemble = (y_pre_test[:,i] - Y_validation[:,i])
            # Fh.append(np.mean(error_ensemble) + np.var(error_ensemble))
            MSE1 = MSE(y_pre_train[:,i],Y_train[:,i])   #训练集均方误差
            MSE2 = MSE(y_pre_test[:, i],  Y_test[:, i])  # 测试集均方误差
            #MSE2 = MSE(y_pre_test[:,i],Y_validation[:,i])      #测试集均方误差
            MAE1 = MAE(y_pre_train[:,i],Y_train[:,i])          #训练集平均绝对误差
            MAE2 = MAE(y_pre_test[:, i], Y_test[:, i])  # 测试集均方误差
            #MAE2 = MAE(y_pre_test[:,i],Y_validation[:,i])      #测试集均方误差
            R21 = r2_score(y_pre_train[:,i],Y_train[:,i])
            R22 = r2_score(y_pre_test[:, i],  Y_test[:, i])
            #R22 = r2_score(y_pre_test[:, i], Y_validation[:, i])
            MAE1 = MAE(y_pre_train[:, i], Y_train[:, i])  # 训练集平均绝对误差
            MAE2 = MAE(y_pre_test[:, i],  Y_test[:, i])
            #MAE2 = MAE(y_pre_test[:, i], Y_validation[:, i])
            PED1 = PED(y_pre_train[:, i], Y_train[:, i],detal=0.05)  # 训练集平均绝对误差
            PED2 = PED(y_pre_test[:, i], Y_test[:, i], detal=0.15)
            #PED2 = PED(y_pre_test[:, i], Y_validation[:, i],detal=0.15)
            MSE_train[j,i] = MSE1
            MSE_test[j,i] = MSE2
            MAE_train[j,i] = MAE1
            MAE_test[j,i] = MAE2
            R2_train[j,i] = R21
            R2_test[j,i] = R22
            PED_train[j, i] = PED1
            PED_test[j, i] = PED2
            # print('训练集{}:MSE{}\n,测试集{}:MSE:{}\n'.format(index[i],MSE1,index[i],MSE2))
            # MAE_train = mean_absolute_error(y_pre_train,Y_train)
            # MAE_test = mean_absolute_error(y_pre_test,Y_validation)
        # mse[0].append(MSE1), mse[1].append(MSE2)
        # mae2[0].append(MAE_train), mae2[1].append(MAE_test)
        # PED1[0].append(PED(y_pre_train,Y_train)),PED1[1].append(PED(y_pre_test,Y_validation))
        j = j + 1
    train = pd.DataFrame(MSE_train, columns=[index[:6]], index=key_label)
    test = pd.DataFrame(MSE_test,columns=[index[:6]],index=key_label)
    print('训练集MSE\n',train)
    print('测试集MSE\n',test)
    #train = pd.DataFrame(MAE_train, columns=[index[:6]], index=key_label)
    #test = pd.DataFrame(MAE_test,columns=[index[:6]],index=key_label)
    #print('训练集MAE\n',train)
    #print('测试集MAE\n',test)
    train = pd.DataFrame(R2_train, columns=[index[:6]], index=key_label)
    test = pd.DataFrame(R2_test, columns=[index[:6]], index=key_label)
    print('训练集R2\n', train)
    print('测试集R2\n', test)
    #train = pd.DataFrame(PED_train, columns=[index[:6]], index=key_label)
    #test = pd.DataFrame(PED_test, columns=[index[:6]], index=key_label)
    #print('训练集PED\n', train)
    #print('测试集PED\n', test)

#plt.figure()
#for i in range(6):
   # plt.plot(train.iloc[:,i],label=index[i])
#plt.legend()
#plt.show()

#y_pre_test_1 = pd.DataFrame(y_pre_test)
#y_pre_test_1.to_excel(r'D:\pycharm项目\2021辛烷值/y_pre_test.xlsx', index=False)
#Y_test = pd.DataFrame(Y_test)
#Y_test.to_excel(r'D:\pycharm项目\2021辛烷值/Y_test.xlsx', index=False)
#X_train_1 = pd.DataFrame( X_train)
#X_train_1.to_excel(r'D:\pycharm项目\2021辛烷值/X_train.xlsx', index=False)
#X_validation = pd.DataFrame(X_validation)
#X_validation.to_excel(r'D:\pycharm项目\2021辛烷值/X_validation.xlsx', index=False)
#Y_train_1 = pd.DataFrame( Y_train)
#Y_train_1.to_excel(r'D:\pycharm项目\2021辛烷值/ Y_train.xlsx', index=False)
#X_train = pd.DataFrame(X_train)
#X_train.to_excel(r'D:\pycharm项目\2021辛烷值/ X_train.xlsx', index=False)

#est = model.estimators_[0]
#feature_importances = pd.DataFrame(est.feature_importances_,
                                 # columns=['importance'],index =['加氢汽油', 'MTBE', '车用异辛烷', '乙苯', '研究法辛烷值', '抗爆性', '10%蒸发温度', '50%蒸发温度',  '硫含量', '苯含量', '烯烃含量', '芳烃含量',  '密度']).sort_values('importance')#print(feature_importances)
#feature_importances.plot(kind = 'barh')
#plt.show()
#feature_importances = pd.DataFrame(est.feature_importances_,
                                   #columns=['importance'],index =['加氢汽油', '醚化汽油', 'MTBE', '车用异辛烷', '汽油重芳烃', '生成油','乙苯', '甲苯', '二甲苯' ]).sort_values('importance')
#print(feature_importances)
#feature_importances.plot(kind = 'barh')
#plt.show()
#fig,ax = plt.subplots(figsize=(12,6))
#plot_importance(est,
                #height=0.5,
                #ax=ax,
                #max_num_features=9)
#plt.show()00.25
#shap.initjs()
#explainer = shap.TreeExplainer(est)
#shap_values = explainer.shap_values(X_train)
#features = pd.DataFrame(X_train) #将numpy的array数组x_test转为dataframe格式。
#features.columns = ['加氢汽油', '醚化汽油', 'MTBE', '车用异辛烷', '汽油重芳烃','生成油', '乙苯', '甲苯', '二甲苯' ] #添加特征名称
#shap.summary_plot(shap_values, X_test)
#shap.summary_plot(shap_values, features, X_train)
#features=['加氢汽油', '醚化汽油', 'MTBE', '车用异辛烷', '汽油重芳烃','生成油' ,'乙苯', '甲苯', '二甲苯']
#shap.summary_plot(shap_values,  data[features],max_display=9)
#shap.summary_plot(shap_values, features,max_display=9,show=False)
# mse1 = np.array(mse[0]).reshape(10,8)
# mse2 = np.array(mse[1]).reshape(10,8)
# MAE_train = np.array(mae2[0]).reshape(10,8)
# MAE_test = np.array(mae2[1]).reshape(10,8)
# PED_train = np.array(PED1[0]).reshape(10,8)
# PED_test  = np.array(PED1[1]).reshape(10,8)
# Fh  = np.array(Fh).reshape(10,8)
# print(np.mean(Fh,axis=0))
#
# mse_train1 = pd.DataFrame(mse1,columns=key_label,index=num_label)
# mse_test1 = pd.DataFrame(mse2,columns=key_label,index=num_label)
# MAE_train1 = pd.DataFrame(MAE_train,columns=key_label,index=num_label)
# MAE_test1 = pd.DataFrame(MAE_test,columns=key_label,index=num_label)
# PED_train1 = pd.DataFrame(PED_train,columns=key_label,index=num_label)
# PED_test1 = pd.DataFrame(PED_test,columns=key_label,index=num_label)
# # print(mse_train1)
# # print(mse_test1)
# # print(MAE_train1)
# # print(MAE_test1)
# # print(PED_train1)
# # print(PED_test1)
#
# '''平均值'''
# mse_train = mean(mse_train1)
# mse_test = mean(mse_test1)
# mse  = pd.concat([mse_train,mse_test],axis=1)
# mse.columns = ['mse_train','mse_test']
# print('mse\n{}'.format(mse.T))
#
# MAE_train = mean(MAE_train1)
# MAE_test = mean(MAE_test1)
# mae_test1= pd.concat([MAE_train,MAE_test],axis=1)
# mae_test1.columns=['train','test']
# print('mae_test1\n{}'.format(mae_test1.T))
#
# PED1_train = mean(PED_train1)
# PED1_test = mean(PED_test1)
# PED_test1 = pd.concat([PED1_train,PED1_test],axis=1)
# PED_test1.columns = ['train','test']
# print('PED_test1\n{}'.format(PED_test1.T))
#
# # 折线图
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(mse_train,'-*',label = '训练集')
# ax.plot(mse_test,'-s',label = '测试集')
# x_ticks = ax.set_xticks([i for i in range(len(key_label))])
# x_labels = ax.set_xticklabels(key_label,rotation=45,fontdict=font1)
# ax.set_title("多算法10次mse平均值",fontdict=font2)
# ax.set_xlabel("算法",fontdict=font2)
# ax.set_ylabel("MSE均值",fontdict=font2)
# plt.tick_params(labelsize=12)
# labels = ax.get_xticklabels() + ax.get_yticklabels()
# [label.set_fontname('Times New Roman') for label in labels]
# # #     y_ticks=ax.set_yticks([])
# # #     y_labels=ax.set_yticklabels([-20+i for i in range(20)],rotation=0,fontsize=14)
# plt.grid()
# plt.savefig('../final_fig/辛烷值归一化MSE均值比较.jpg',dpi=500,bbox_inches = 'tight')#保存图片
# plt.legend()
# plt.show()
#
# # 折线图
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(MAE_train,'-*',label = '训练集')
# ax.plot(MAE_test,'-s',label = '测试集')
# x_ticks = ax.set_xticks([i for i in range(len(key_label))])
# x_labels = ax.set_xticklabels(key_label,rotation=45,fontdict=font1)
# ax.set_title("多算法10次MAE平均值",fontdict=font2)
# ax.set_xlabel("算法",fontdict=font2)
# ax.set_ylabel("MAE均值",fontdict=font2)
# plt.tick_params(labelsize=12)
# labels = ax.get_xticklabels() + ax.get_yticklabels()
# [label.set_fontname('Times New Roman') for label in labels]
# # #     y_ticks=ax.set_yticks([])
# # #     y_labels=ax.set_yticklabels([-20+i for i in range(20)],rotation=0,fontsize=14)
# plt.grid()
# plt.savefig('../final_fig/辛烷值归一化MSE均值比较.jpg',dpi=500,bbox_inches = 'tight')#保存图片
# plt.legend()
# plt.show()
#
#
# # 折线图
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(PED1_train,'-*',label = '训练集')
# ax.plot(PED1_test,'-s',label = '测试集')
# x_ticks = ax.set_xticks([i for i in range(len(key_label))])
# x_labels = ax.set_xticklabels(key_label,rotation=45,fontdict=font1)
# ax.set_title("多算法10次PED平均值",fontdict=font2)
# ax.set_xlabel("算法",fontdict=font2)
# ax.set_ylabel("PED均值",fontdict=font2)
# plt.tick_params(labelsize=12)
# labels = ax.get_xticklabels() + ax.get_yticklabels()
# [label.set_fontname('Times New Roman') for label in labels]
# # #     y_ticks=ax.set_yticks([])
# # #     y_labels=ax.set_yticklabels([-20+i for i in range(20)],rotation=0,fontsize=14)
# plt.grid()
# plt.savefig('../final_fig/辛烷值归一化MSE均值比较.jpg',dpi=500,bbox_inches = 'tight')#保存图片
# plt.legend()
# plt.show()
#
#
#
