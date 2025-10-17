import pandas as pd
import numpy as np
from pathlib import Path

THIS_DIR = Path(r"C:\Users\reons\Documents\Certificates\Projects\Brain Stroke Prediction\stroke_prediction_dataset.csv").parent

FILE_NAME = "stroke_prediction_dataset.csv"

DATA_PATH = THIS_DIR / FILE_NAME

try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print(f"Error: The file '{FILE_NAME}' was not found at the expected location: {DATA_PATH}")
    raise


print(df)

df.info()

df.isna().sum()
df.info()

df.drop(['Patient Name', 'Residence Type', 'Family History of Stroke', 'Dietary Habits', 'Stress Levels', 'Symptoms'], axis=1, inplace=True)

object_cols = list(df.select_dtypes(include=['category','object']))
print(object_cols)

#Extract numerical Blood Pressure (BP)
df[['Systolic_BP', 'Diastolic_BP']] = df['Blood Pressure Levels'].str.split('/', expand=True).astype(float)
df.info()

#Extract numerical Cholesterol levels as ldl and hdl
df['LDL'] = df['Cholesterol Levels'].str.extract(r'LDL: (\d+)').astype(float)
df['HDL'] = df['Cholesterol Levels'].str.extract(r'HDL: (\d+)').astype(float)

#Convert diagnosis to numerical
df['Stroke'] = df['Diagnosis'].apply(lambda x: 1 if x == 'Stroke' else 0)

#Drop original columns
df.drop(columns = ['Patient ID', 'Blood Pressure Levels', 'Cholesterol Levels', 'Diagnosis'], inplace=True)
df.info()

#One hot encoder
columns_to_encode = [
    'Gender',
    'Marital Status',
    'Work Type',
    'Smoking Status',
    'Alcohol Intake',
    'Physical Activity'
]

#get_dummies is the process of converting categorical variables into numerical dummy variables
#drop_first is aparameter that is commonly used in data processing when creating dummy variables, the purpose is to reduce redundancy and simplify the model
df_encoded = pd.get_dummies(df, columns=columns_to_encode, drop_first=True)
df_encoded.head(10)

#Method 1: Splitting data into training and testing datasets using iloc
from sklearn.model_selection import train_test_split
df_X = df_encoded.iloc[:,[0,1,2,3]]
df_y = df_encoded['Stroke']

#Method 2: Splitting data into training and testing datasets using drop
#"Drop" here is used to drop "Stroke" column therefore separating the input data from target data
#df_X = df_encoded.drop('Stroke', axis=1)
#df_y = df_encoded['Stroke']

X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.2, random_state=42)

X_train.info()
y_train
X_test.info()
y_test

#Feature Selection
cr = df_encoded.corr()

import seaborn as sns 
#Seaborn is a high level interface for creating statistical graphics

#Creating Heatmap for correaltion
sns.heatmap(cr,annot=True,cmap="coolwarm")

from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(n_neighbors=5)
#Train the model using training sets
knn.fit(X_train, y_train)
knn.fit(X_test, y_test)

#Predicting Stroke for test dataset using knn
y_pred = knn.predict(X_test)
result = pd.DataFrame({'y_actual':y_test,'y_predict':y_pred})
print(result)

#Evaluate your model using RMSE(Root mean square error)
import math

math.sqrt(((y_test-y_pred)**2).mean())

#Analyse model performance visually
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

sorted_result = result.sort_values(by='y_actual')
x1 = np.arange(1,len(y_test)+1)
#np.arange: It generates arrays with evenly spaced values within the given intervals
#note for 1: It is single dimensional array that contains a sequence of numbers and this numbering will start from 1 and will increment by 1 unit(+1)
#note for len(y_test): it gets the number of elements that are available in y_test
y1 = result.y_actual
y2 = result.y_predict

figure(figsize=(10, 8))
plt.plot(x1, y1)
plt.show()

figure(figsize=(10, 8))
plt.plot(x1, y2)
plt.show()

#Comparing plots for (x1,y1) and (x2,y2)
figure(figsize=(10, 8))
plt.plot(x1, y1, x1, y2)
plt.show()


#Analyze model performance visually
import matplotlib.cm as cm
colors1 = cm.rainbow(y_test*500)
colors2= cm.rainbow(y_pred*500)
figure(figsize=(10, 8))
plt.plot(x1, y1)
plt.show()

#Analyzing the error patterns
figure(figsize=(15, 12))
plt.plot(x1, y1-y2)
plt.show()
