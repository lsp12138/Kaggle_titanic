#!/usr/bin/env python
# encoding: utf-8

"""
@version: 
@author: 
@time: 2016/12/24 23:37
@remark:
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


def sub():
    x_train = pd.read_csv(r'E:\kaggle\titanicv2\cleaned_data\cleaned_train_feature.csv', header=None).values
    y_train = pd.read_csv(r'E:\kaggle\titanicv2\cleaned_data\train_tag.csv', header=None).values
    x_test = pd.read_csv(r'E:\kaggle\titanicv2\cleaned_data\cleaned_test_feature.csv', header=None).values

    #集合方法
    clf1 = svm.SVC(probability=True,random_state=7)
    clf2 = RandomForestClassifier(random_state=7)
    clf3 = GradientBoostingClassifier(random_state=7)
    voting_class = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('dt', clf3)], voting='soft',
                                    weights=[1, 1, 1])
    vote = voting_class.fit(x_train, y_train)
    #保存模型
    #joblib.dump(vote,r'E:\kaggle\titanic\model\vote_sub.model')
    #vote = joblib.load(r'E:\kaggle\titanicv2\model\0.850746268657vote1.model')
    y_test_pred = vote.predict(x_test)
    pre = pd.DataFrame(y_test_pred,index=None,columns=['Survived'])
    pre.to_csv(r"E:\kaggle\titanicv2\sub\pre7.csv",index=None)
    df = pd.read_csv(r'E:\kaggle\titanic\data\test.csv')
    dfID = df.PassengerId
    sub = pd.concat([dfID,pre],axis=1)
    sub.to_csv(r"E:\kaggle\titanicv2\sub\sub7.csv", index=None)
    print 'ok'


if __name__ == '__main__':
    sub()
