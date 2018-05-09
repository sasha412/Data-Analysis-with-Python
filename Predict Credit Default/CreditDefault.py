
# coding: utf-8

"""
Created on Sat Mar  3 20:36:34 2018

@authors: Sashank

"""
# =============================================================================
# #import Class Libraries
# =============================================================================

# class for logistic regression
from sklearn.linear_model import LogisticRegression

# class for decision tree
from Class_tree import DecisionTree
from sklearn.tree import DecisionTreeClassifier
#class for neural network
from Class_FNN import NeuralNetwork
from sklearn.neural_network import MLPClassifier
#class for random forest
from sklearn.ensemble import RandomForestClassifier
#other needed classes
from Class_replace_impute_encode import ReplaceImputeEncode
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
# show plots
import seaborn as sns


# =============================================================================
# # Functions Used 
# =============================================================================
# Get classification metrics
def getClassificationMetrics(tn, fp, fn, tp, y_test, pred_test):
       
    #sensitivity
    Recall=tp/(tp+fn);
    print("sensitivity: ", Recall)
    #specificity
    Specificity= tn/(tn+fp)
    print("specificity: ", Specificity)
    #accuracy
    print("accuracy: ", (tp+tn)/(tp+fn+tn+fp))
    #precision
    Precision=tp/(tp+fp);
    print("precision: ", Precision)
    #f1 score
    print("f1_score: ",  (2*Recall*Precision)/(Recall + Precision))
    #misclassification
    print("Misc: ", (fp+fn)/(tp+fn+tn+fp))
    #False Positive Rate
    print("FPR: ", 1-Specificity)
        
    return


# =============================================================================
# #Get Data from excel file
# =============================================================================

file_path = "/Users/sasha/Library/Mobile Documents/com~apple~CloudDocs/STAT 656/Midterm/CreditCard_Defaults.xlsx"
df = pd.read_excel(file_path)
df.head()

# create attribute map
# Attribute Map:  the key is the name in the DataFrame
# The first number of 0=Interval, 1=binary and 2=nomial
# The 1st tuple for interval attributes is their lower and upper bounds
# The 1st tuple for categorical attributes is their allowed categories
# The 2nd tuple contains the number missing and number of outliers
attribute_map = {
    "Default":[1,(0,1),[0,0]],
   "Gender":[1,(1,2),[0,0]],
  "Education":[2,(0,1,2,3,4,5,6),[0,0]],
   "Marital_Status":[2,(0,1,2,3),[0,0]],
    "card_class":[2,(1,2,3),[0,0]],
    "Age":[0,(20,80),[0,0]],
    "Credit_Limit":[0,(100,80000),[0,0]],
    "Jun_Status":[0,(-2,8),[0,0]],
   "May_Status":[0,(-2,8),[0,0]],
   "Apr_Status":[0,(-2,8),[0,0]],
    "Mar_Status":[0,(-2,8),[0,0]],
    "Feb_Status":[0,(-2,8),[0,0]],
    "Jan_Status":[0,(-2,8),[0,0]],
     "Jun_Bill":[0,(-12000,32000),[0,0]],
   "May_Bill":[0,(-12000,32000),[0,0]],
    "Apr_Bill":[0,(-12000,32000),[0,0]],
    "Mar_Bill":[0,(-12000,32000),[0,0]],
    "Feb_Bill":[0,(-12000,32000),[0,0]],
    "Jan_Bill":[0,(-12000,32000),[0,0]],
    "Jun_Payment":[0,(0,60000),[0,0]],
    "May_Payment":[0,(0,60000),[0,0]],
    "Apr_Payment":[0,(0,60000),[0,0]],
    "Mar_Payment":[0,(0,60000),[0,0]],
    "Feb_Payment":[0,(0,60000),[0,0]],
    "Jan_Payment":[0,(0,60000),[0,0]],
    "Jun_PayPercent":[0,(0,1),[0,0]],
    "May_PayPercent":[0,(0,1),[0,0]],
    "Apr_PayPercent":[0,(0,1),[0,0]],
    "Mar_PayPercent":[0,(0,1),[0,0]],
    "Feb_PayPercent":[0,(0,1),[0,0]],
    "Jan_PayPercent":[0,(0,1),[0,0]]   
    
}

df = df.drop("Customer", axis=1)
# drop=False - used for Decision tree
rie_1 = ReplaceImputeEncode(data_map=attribute_map, nominal_encoding='one-hot', interval_scale = 'std',drop = False, display=True)
encoded_df = rie_1.fit_transform(df)
#create X and y
y = encoded_df["Default"]
X = encoded_df.drop("Default", axis=1)
np_y = np.ravel(y)



# =============================================================================
# #Initial Analysis
# =============================================================================
#Correlation Plot of entire dataset
# Deductions: All the columns indicating customers bills are correlated 
#All columns indicating months behind payement are correlated
#

corr = df.corr()
g=sns.heatmap(corr, 
            xticklabels="auto",
            yticklabels="auto")
g.set_yticklabels(g.get_yticklabels(), rotation = 0, fontsize = 8)
g.set_xticklabels(g.get_yticklabels(), rotation = 70, fontsize = 6)



# =============================================================================
# # Hyper Parameter Optimization with 10-fold Cross Validation 
# =============================================================================

#Cross Validation for Logistic Regression

#Do Forward stepwise regression with kfold cross validation      
remaining = set(X.columns)
response='Default'
selected = []
current_score, best_new_score = 0.0, 0.0
while remaining and current_score == best_new_score:
    scores_with_candidates = []
    for candidate in remaining:   
        #forward step wise columns
        formula = selected + [candidate]
        
        X_Forward= encoded_df[formula]   
        model = LogisticRegression()
        kf = KFold(n_splits=10,random_state=12345)
        scoring = 'f1'
        results = cross_val_score(model, X_Forward, np_y, cv=kf, scoring=scoring)
        score= results.mean()       
        scores_with_candidates.append((score, candidate))
        
    scores_with_candidates.sort()  
    best_new_score, best_candidate = scores_with_candidates.pop()
    if(best_new_score==current_score):
        break;
    if current_score < best_new_score:
        print(best_new_score,current_score)
        remaining.remove(best_candidate)
        selected.append(best_candidate)
        print(selected)
        current_score = best_new_score

# =============================================================================
# #Output
# 0.315606776734 0.0
# ['Jun_Status']
# 0.461891482261 0.315606776734
# ['Jun_Status', 'card_class2']
# 0.463242620061 0.461891482261
# ['Jun_Status', 'card_class2', 'Marital_Status3']
# 0.46343712211 0.463242620061
# ['Jun_Status', 'card_class2', 'Marital_Status3', 'Education6']
# 0.463632243445 0.46343712211
# ['Jun_Status', 'card_class2', 'Marital_Status3', 'Education6', 'Education5']        
# 
# =============================================================================


#Best model is at ['Jun_Status', 'card_class2', 'Marital_Status3', 'Education6', 'Education5']
        
        


# =============================================================================
# # Cross Validation for decision tree: 
# =============================================================================
'''
The results show that max_depth has the most influence while min_samples_split won't much have inflence on metrics score. 
When max_depth is below 20, the metrics score do slightly increase when
the min_samples_leaf increases. When the max_depth is bigger than 20, then actually increase of 
min_samples_leaf reduces the metrics score. Since there isn't much difference between depth 20,25,50, I stop increase
this number.We can try to use Maximum Tree Depth:25, Min_samples_leaf:3, Min_samples_split:5 or Maximum Tree Depth:6, Min_samples_leaf:7,
Min_samples_split:5 in the 70/30 validation.
'''
depth_list = [6]
#[5,6,7,8,10, 12, 15, 20, 25, 50]
minSamplesLeaf= [3,5,7]
minSamplesSplit=[3,5,7]


score_list = ['accuracy', 'recall', 'precision', 'f1']
for d in depth_list:
    for l in minSamplesLeaf:
        for s in minSamplesSplit:
            print("\nMaximum Tree Depth: ", d, "Min_samples_leaf", l, "Min_samples_split", s)
            dtc = DecisionTreeClassifier(max_depth=d, min_samples_leaf=l,  min_samples_split=s, random_state=12345)
            dtc = dtc.fit(X,np_y)
            scores = cross_validate(dtc, X, np_y, scoring=score_list, return_train_score=False, cv=10)
    
            print("{:.<13s}{:>6s}{:>13s}".format("Metric", "Mean", "Std. Dev."))
            for s in score_list:
                var = "test_"+s
                mean = scores[var].mean()
                std  = scores[var].std()
                print("{:.<13s}{:>7.4f}{:>10.4f}".format(s, mean, std))
                

#Maximum Tree Depth:  25 Min_samples_leaf 3 Min_samples_split 5
#Metric.......  Mean    Std. Dev.
#accuracy..... 0.8296    0.0108
#recall....... 0.4660    0.0241
#precision.... 0.4745    0.0336
#f1........... 0.4697    0.0241

#Maximum Tree Depth:  6 Min_samples_leaf 7 Min_samples_split 5
#Metric.......  Mean    Std. Dev.
#accuracy..... 0.8751    0.0072
#recall....... 0.4524    0.0389
#precision.... 0.6701    0.0385
#f1........... 0.5390    0.0319





# =============================================================================
# # Cross-Validation for neural network: There isn't much difference going on with different 
# # layers/perceptrons. Best Neural Network is (7,6)
# =============================================================================
network_list = [(3), (11), (5,4), (6,5), (7,6), (8,7),(9,8)]
score_list = ['accuracy', 'recall', 'precision', 'f1']
for nn in network_list:
    print("\nNetwork: ", nn)
    fnn = MLPClassifier(hidden_layer_sizes=nn, activation='logistic',                     solver='lbfgs', max_iter=1000, random_state=12345)
    fnn = fnn.fit(X,np_y)
    scores = cross_validate(fnn, X, np_y, scoring=score_list,                             return_train_score=False, cv=10)
    
    print("{:.<13s}{:>6s}{:>13s}".format("Metric", "Mean", "Std. Dev."))
    for s in score_list:
        var = "test_"+s
        mean = scores[var].mean()
        std  = scores[var].std()
        print("{:.<13s}{:>7.4f}{:>10.4f}".format(s, mean, std))
#Network:  (7, 6)
# =============================================================================
# Metric.......  Mean    Std. Dev.
# accuracy..... 0.8695    0.0079
# recall....... 0.4537    0.0268
# precision.... 0.6382    0.0456
# f1........... 0.5293    0.0240
# =============================================================================




# =============================================================================
# # Cross-Validation for random forest
# =============================================================================
'''
Metrics scores increase when we enlarge the n_estimators and max_features. However, when we 
further increase the n_estimators to 70, the metrics scores didn't show much difference from the 
one with 50 estimators. So we stop it to increase more. "auto" in max_feature usually gives 
good results, but it's not the case here. Finally, we would use the model with Number of Trees:70,
Max_features:0.5 for 70/30 split validation 
'''
estimators_list   = [10, 30, 50, 70]
max_features_list = ['auto', 0.5, 0.7]
score_list = ['accuracy', 'recall', 'precision', 'f1']
max_f1 = 0
for e in estimators_list:
    for f in max_features_list:
        print("\nNumber of Trees: ", e, " Max_features: ", f)
        rfc = RandomForestClassifier(n_estimators=e, criterion="gini", max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=f,                     n_jobs=1, bootstrap=True, random_state=12345)
        rfc= rfc.fit(X, np_y)
        scores = cross_validate(rfc, X, np_y, scoring=score_list, return_train_score=False, cv=10)
        
        print("{:.<13s}{:>6s}{:>13s}".format("Metric", "Mean", "Std. Dev."))
        for s in score_list:
            var = "test_"+s
            mean = scores[var].mean()
            std  = scores[var].std()
            print("{:.<13s}{:>7.4f}{:>10.4f}".format(s, mean, std))


#Number of Trees:  70  Max_features:  0.5
#Metric.......  Mean    Std. Dev.
#accuracy..... 0.8832    0.0062
#recall....... 0.4706    0.0367
#precision.... 0.7109    0.0330
#f1........... 0.5653    0.0287     




# =============================================================================
# # Evaluate the best model:
# =============================================================================
'''
Decision tree with(Maximum Tree Depth: 25 Min_samples_leaf:3 Min_samples_split: 5) is overfitting
the data. So does the random forest (n_estimator:70, max_feature:0.5). The best model is neural
network (7,6), which scores higher across all metrics. Please refer report for further details. 

'''
   

#Best Model forward step wise selection - 0.42 model score

X_ForwardStepWise=X[['Jun_Status', 'card_class2', 'Marital_Status3', 'Education6', 'Education5']]        

#Split Data 70-30 
#split 70, 30 and use X_train and y_train for doing 70/30 cross validation
X_train, X_test, y_train, y_test = train_test_split(X_ForwardStepWise,np_y,test_size = 0.3, random_state=12345)

#train model 
logModel= LogisticRegression()
logModel.fit(X_train, y_train)

#test model
pred_test = logModel.predict(X_test)
pred_train = logModel.predict(X_train)
#get Metrics
tn, fp, fn, tp = confusion_matrix(y_test, pred_test).ravel()
tn_train, fp_train, fn_train, tp_train = confusion_matrix(y_train,pred_train ).ravel()
(tn, fp, fn, tp)

getClassificationMetrics(tn,fp,fn,tp,y_test,pred_test) 
getClassificationMetrics(tn_train, fp_train, fn_train, tp_train,y_train,pred_train)       
            

# =============================================================================
# sensitivity_test/recall_test/TPR_test: 0.420097697139
# specificity_test:  0.961675697106
# accuracy_test:  0.875444444444
# precision_test:  0.674887892377
# f1_score:  0.517849462366
# misc_test:  0.124555555556
# FPR:  0.0383243028941
# =============================================================================

            

X_train, X_validate, y_train, y_validate =  train_test_split(X, np_y,test_size = 0.3, random_state=12345)


#Decison Tree
dtc = DecisionTreeClassifier(max_depth=25, min_samples_leaf=3, min_samples_split=5, random_state=12345)
dtc = dtc.fit(X_train,y_train)
DecisionTree.display_binary_split_metrics(dtc, X_train, y_train,X_validate, y_validate)
features = list(X)
classes = ['No-Default', 'Default']

DecisionTree.display_importance(dtc, features)


# =============================================================================
# #Output
# Model Metrics..........       Training     Validation
# Observations...........          21000           9000
# Features...............             41             41
# Maximum Tree Depth.....             25             25
# Minimum Leaf Size......              3              3
# Minimum split Size.....              5              5
# Mean Absolute Error....         0.0552         0.1807
# Avg Squared Error......         0.0276         0.1502
# Accuracy...............         0.9580         0.8284
# Precision..............         0.8924         0.4612
# Recall (Sensitivity)...         0.8439         0.4599
# F1-score...............         0.8675         0.4605
# MISC (Misclassification)...       4.2%          17.2%
#      class 0...............       2.0%          10.2%
#      class 1...............      15.6%          54.0%
# 
# 
# Training
# Confusion Matrix  Class 0   Class 1  
# Class 0.....     17231       348
# Class 1.....       534      2887
# 
# 
# Validation
# Confusion Matrix  Class 0   Class 1  
# Class 0.....      6797       770
# Class 1.....       774       659
# 
# FEATURE......... IMPORTANCE
# Jun_Status......   0.2742
# May_Status......   0.0498
# Credit_Limit....   0.0490
# Age.............   0.0454
# Jun_Bill........   0.0395
# Jan_Bill........   0.0374
# May_Bill........   0.0315
# May_PayPercent..   0.0306
# Apr_Bill........   0.0289
# Jan_Payment.....   0.0287
# May_Payment.....   0.0287
# Apr_PayPercent..   0.0282
# Jun_PayPercent..   0.0279
# Mar_Bill........   0.0276
# Apr_Payment.....   0.0275
# Mar_PayPercent..   0.0273
# Feb_Bill........   0.0271
# Jun_Payment.....   0.0252
# Feb_PayPercent..   0.0243
# Jan_PayPercent..   0.0230
# Feb_Payment.....   0.0225
# Mar_Payment.....   0.0219
# Mar_Status......   0.0164
# Jan_Status......   0.0153
# Apr_Status......   0.0087
# Gender..........   0.0074
# Education2......   0.0059
# Feb_Status......   0.0043
# Education3......   0.0042
# Education1......   0.0041
# Marital_Status2.   0.0028
# Marital_Status1.   0.0023
# card_class1.....   0.0009
# card_class0.....   0.0008
# card_class2.....   0.0004
# Education5......   0.0003
# Marital_Status3.   0.0001
# Education0......   0.0000
# Education4......   0.0000
# Education6......   0.0000
# Marital_Status0.   0.0000
# =============================================================================


#Neural Network
fnn = MLPClassifier(hidden_layer_sizes=(7,6), activation='logistic', solver='lbfgs', max_iter=1000, random_state=12345)
fnn = fnn.fit(X_validate,y_validate)
NeuralNetwork.display_binary_split_metrics(fnn, X_train, y_train, X_validate, y_validate)
# =============================================================================
# #output
# Model Metrics..........       Training     Validation
# Observations...........          21000           9000
# Features...............             41             41
# Number of Layers.......              2              2
# Number of Outputs......              1              1
# Number of Weights......            349            349
# Activation Function....       logistic       logistic
# Mean Absolute Error....         0.1913         0.1602
# Avg Squared Error......         0.1111         0.0798
# Accuracy...............         0.8602         0.8931
# Precision..............         0.6065         0.7367
# Recall (Sensitivity)...         0.4046         0.5115
# F1-score...............         0.4854         0.6038
# MISC (Misclassification)...      14.0%          10.7%
#      class 0...............       5.1%           3.5%
#      class 1...............      59.5%          48.8%
# 
# 
# Training
# Confusion Matrix  Class 0   Class 1  
# Class 0.....     16681       898
# Class 1.....      2037      1384
# 
# 
# Validation
# Confusion Matrix  Class 0   Class 1  
# Class 0.....      7305       262
# Class 1.....       700       733
# =============================================================================


#Random Forest
rfc = RandomForestClassifier(n_estimators=70, criterion="gini", max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=0.5, n_jobs=1, bootstrap=True, random_state=12345)
rfc= rfc.fit(X_train, y_train)
DecisionTree.display_importance(rfc, features) 
rfc_validate = rfc.predict(X_validate)
rfc_train = rfc.predict(X_train)
tn, fp, fn, tp = confusion_matrix(y_validate, rfc_validate).ravel()
tn_t, fp_t, fn_t, tp_t = confusion_matrix(y_train, rfc_train).ravel()


getClassificationMetrics(tn,fp,fn,tp,y_validate,rfc_validate)
getClassificationMetrics(tn_t,fp_t,fn_t,tp_t,y_train,rfc_train)

#Output
# =============================================================================
# FEATURE......... IMPORTANCE
# Jun_Status......   0.1649
# May_Status......   0.0750
# Age.............   0.0490
# Jun_Bill........   0.0416
# Credit_Limit....   0.0410
# Jan_Bill........   0.0350
# May_Bill........   0.0347
# Apr_Payment.....   0.0346
# Jan_Payment.....   0.0328
# Feb_Bill........   0.0320
# Mar_Bill........   0.0309
# Apr_Bill........   0.0308
# Jun_Payment.....   0.0307
# Apr_PayPercent..   0.0294
# Jun_PayPercent..   0.0290
# May_PayPercent..   0.0290
# May_Payment.....   0.0287
# Mar_Payment.....   0.0284
# Feb_Payment.....   0.0272
# Mar_PayPercent..   0.0270
# Jan_PayPercent..   0.0269
# Feb_PayPercent..   0.0258
# Apr_Status......   0.0233
# Feb_Status......   0.0167
# Mar_Status......   0.0164
# Jan_Status......   0.0129
# Gender..........   0.0077
# Education2......   0.0059
# Marital_Status2.   0.0058
# Education1......   0.0055
# Education3......   0.0054
# Marital_Status1.   0.0052
# card_class1.....   0.0033
# card_class0.....   0.0027
# card_class2.....   0.0018
# Marital_Status3.   0.0017
# Education5......   0.0007
# Marital_Status0.   0.0003
# Education4......   0.0002
# Education6......   0.0002
# Education0......   0.0000
# 
# for validation data
# sensitivity:  0.471039776692
# specificity:  0.962336460949
# accuracy:  0.884111111111
# precision:  0.703125
# f1_score:  0.564145424154
# Misc:  0.115888888889
# FPR:  0.0376635390511
# 
# for train data
# sensitivity:  0.999415375621
# specificity:  1.0
# accuracy:  0.999904761905
# precision:  1.0
# f1_score:  0.999707602339
# Misc:  9.52380952381e-05
# FPR:  0.0
# =============================================================================



