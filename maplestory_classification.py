import cv2
import numpy as np 
import os
import pandas as pd
from skimage.io import imread
from skimage.transform import resize
from skimage import io
import matplotlib.pyplot as plt 



input_dir = 'C:/Users/liang/OneDrive/Desktop/Maplestory-Classifier'  
data = []
labels = []
categories = ['Mushmom_Classification',
              'Poisonous_Mushroom_Classification',
              'Horny_Mushroom_Classification',
              'Green_Mushroom_Classification', 
              'Blue_Mushmom_Classification', 
              'Zombie_Mushmom_Classification'] 



for category_idx , category in enumerate(categories): 
    for file in os.listdir(os.path.join(input_dir,category)):   
        img_path = os.path.join(input_dir,category,file)
        img = imread(img_path)
        img = resize(img,(70,70))        
        data.append(img.flatten())
        labels.append(category) 


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data,labels,test_size= 0.25, random_state=42) 


y_train_label = (np.array(y_train) == 'Mushmom_Classification')
y_test_label = (np.array(y_test) == 'Mushmom_Classification') 

  


from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier 
from sklearn.naive_bayes import GaussianNB 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 





from sklearn.preprocessing import StandardScaler, MinMaxScaler

scalers = [StandardScaler(),MinMaxScaler()]


def test_scaler(dataset,target,scalers,model):
    results = {}
    for scaler in scalers: 
        dataset_scaled = scaler.fit_transform(dataset)
        cross_val_score(model,dataset_scaled,target,cv=5,scoring='accuracy')
        scaler_name = type(scaler).__name__
        if scaler_name not in results:
            results[scaler_name] = cross_val_score(model,dataset_scaled,target,cv=5,scoring='accuracy')
    return results



rand_forest = RandomForestClassifier(random_state=42)
rand_forest_cross_validation_test = test_scaler(X_train,y_train_label,scalers,rand_forest)
print(rand_forest_cross_validation_test)


sgd_clf = SGDClassifier(random_state=42)
sgd_clf_cross_validation_test = test_scaler(X_train,y_train_label,scalers,sgd_clf)
print(sgd_clf_cross_validation_test) 


svm_clf = SVC(random_state=42)  
svm_clf_cross_validation_test = test_scaler(X_train,y_train_label,scalers,svm_clf)
print(svm_clf_cross_validation_test) 

guassian_nb = GaussianNB()
guassian_nb_cross_validation_test = test_scaler(X_train,y_train_label,scalers,guassian_nb)
print(guassian_nb_cross_validation_test)  


log_reg = LogisticRegression(random_state=42,max_iter=4000)  
log_reg_cross_validation_test = test_scaler(X_train,y_train_label,scalers,log_reg)
print(log_reg_cross_validation_test)






