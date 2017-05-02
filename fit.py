#!/usr/bin/env python
# encoding: utf-8

"""
@version: 
@author: 
@time: 2016/12/24 15:41
@remark:


#用pylab画散点图：
import pylab as pl
%pylab inline
pl.scatter(df.index,df['Age'])

"""
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn.ensemble import VotingClassifier
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier


def fit():
    feature = pd.read_csv(r'E:\kaggle\titanicv2\cleaned_data\cleaned_train_feature.csv',header=None).values
    target = pd.read_csv(r'E:\kaggle\titanicv2\cleaned_data\train_tag.csv',header=None).values

    x_train, x_test, y_train, y_test = train_test_split(feature,target,test_size=0.15,random_state=None)

    # 随机森林
    rfc = RandomForestClassifier(n_estimators=3,random_state=2)
    rfc.fit(x_train,y_train)
    print(rfc.score(x_test,y_test))

    y_train_pred = rfc.predict(x_train)
    y_test_pred = rfc.predict(x_test)
    tree_train = accuracy_score(y_train, y_train_pred)
    tree_test = accuracy_score(y_test, y_test_pred)
    print('rfc/test accuracies %.4f/%.4f' % (tree_train, tree_test))

    # feature_importance = rfc.feature_importances_
    # feature_importance = 100.0 * (feature_importance / feature_importance.max())
    # print(feature_importance)
    # cvscores = cross_validation.cross_val_score(rfc, feature, target, cv=5)
    # print(cvscores.mean())

    # svm
    svmMod = svm.SVC(probability=True,random_state=2)
    svmMod.fit(x_train, y_train)
    y_train_pred = svmMod.predict(x_train)
    y_test_pred = svmMod.predict(x_test)
    y_train_pred = svmMod.predict(x_train)
    y_test_pred = svmMod.predict(x_test)
    tree_train = accuracy_score(y_train, y_train_pred)
    tree_test = accuracy_score(y_test, y_test_pred)
    print('SVM train/test accuracies %.4f/%.4f' % (tree_train, tree_test))


    # 使用gbdt来分类
    gbdt = GradientBoostingClassifier(random_state=2)
    gbdt = gbdt.fit(x_train, y_train)
    y_train_pred = gbdt.predict(x_train)
    y_test_pred = gbdt.predict(x_test)
    tree_train = accuracy_score(y_train, y_train_pred)
    tree_test = accuracy_score(y_test, y_test_pred)
    print('gbdt/test accuracies %.4f/%.4f' % (tree_train, tree_test))

    # 集合方法 随机性的模型都会有个random_state来控制随机种子，所以得到一个好模型后记下这个值用来复现模型。
    voting_class = VotingClassifier(estimators=[('rfc', rfc), ('gbdt', gbdt),('svm',svmMod)], voting='soft',
                                    weights=[1, 1, 1])
    vote = voting_class.fit(x_train, y_train)
    y_train_pred = vote.predict(x_train)
    y_test_pred = vote.predict(x_test)
    vote_train = accuracy_score(y_train, y_train_pred)
    vote_test = accuracy_score(y_test, y_test_pred)
    print('Ensemble Classifier train/test accuracies %.4f/%.4f' % (vote_train, vote_test))
    #保存模型 ,compress=3解决了生成一堆文件的问题
    joblib.dump(vote,'E://kaggle//titanicv2//model//' + str(vote_test) + 'vote1.model',compress=3)
    #vote = joblib.load('vote.model')



if __name__ == '__main__':
    fit()