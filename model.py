import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import warnings

warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

PATH = "/Users/sarahmakkar/Desktop/OneDrive/Documents/BCIT/CST 4/COMP4949/A2/"
CSV_DATA = "voice.csv"
df = pd.read_csv(PATH + CSV_DATA)

df['label'].replace(['male', 'female'], [0,1], inplace=True)

dataset = df[['meanfreq', 'sd', 'median', 'Q25', 'IQR', 'sp.ent', 'sfm', 'centroid', 'meanfun','label']]

dfAdjusted = dataset['meanfreq'].clip(lower=0.1105003)
dataset['meanfreq'] = dfAdjusted
dfAdjusted = df['sd'].clip(upper=0.10841)
dataset['sd'] = dfAdjusted
dfAdjusted = dataset['median'].clip(lower=0.108125)
dataset['median'] = dfAdjusted
dfAdjusted = dataset['IQR'].clip(upper=0.2143911)
dataset['IQR'] = dfAdjusted
dfAdjusted = dataset['centroid'].clip(lower=0.1105003)
dataset['centroid'] = dfAdjusted

X = dataset[['sd', 'Q25', 'IQR', 'sp.ent', 'sfm', 'meanfun']]
y = dataset['label']

accuracyList = []
precisionList = []
recallList = []
F1List = []

for i in range(0, 10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
    sc_x = StandardScaler()
    X_train_scaled = sc_x.fit_transform(X_train)  # Fit and transform X.
    X_test_scaled = sc_x.transform(X_test)  # Transform X.

    xgb_boost = XGBClassifier()
    xgb_boost.fit(X_train_scaled, y_train)
    predictions = xgb_boost.predict(X_test_scaled)

    precision = metrics.precision_score(y_test, predictions, average='binary')
    precisionList.append(precision)

    recall = metrics.recall_score(y_test, predictions, average='binary')
    recallList.append(recall)

    accuracy = metrics.accuracy_score(y_test, predictions)
    accuracyList.append(accuracy)

    F1 = metrics.f1_score(y_test, predictions)
    F1List.append(F1)

print("\nAccuracy and Standard Deviation:")
print("Average accuracy: " + str(np.mean(accuracyList)))
print("Accuracy std: " + str(np.std(accuracyList)))

print("\nPrecision and Standard Deviation:")
print("Average precision: " + str(np.mean(precisionList)))
print("Precision std: " + str(np.std(precisionList)))

print("\nRecall and Standard Deviation:")
print("Average recall: " + str(np.mean(recallList)))
print("Recall std: " + str(np.std(recallList)))

print("\nF1 Score and Standard Deviation:")
print("Average F1: " + str(np.mean(F1List)))
print("F1 std: " + str(np.std(F1List)))

# Save the model.
with open('model_pkl', 'wb') as files:
    pickle.dump(xgb_boost, files)

# load saved model
with open('model_pkl' , 'rb') as f:
    loadedModel = pickle.load(f)

# Create a single prediction.
singleSampleDf = pd.DataFrame(columns=['sd', 'Q25', 'IQR', 'sp.ent', 'sfm', 'meanfun'])
sd =  0.064
Q25 = 0.015
IQR = 0.075
spent = 0.893
sfm = 0.492
meanfun = 0.084

voiceData = {'sd': sd, 'Q25': Q25, 'IQR': IQR, 'sp.ent': spent, 'sfm': sfm, 'meanfun': meanfun}
singleSampleDf = pd.concat([singleSampleDf,
                            pd.DataFrame.from_records([voiceData])])

singlePrediction = loadedModel.predict(singleSampleDf)
print("Single prediction: " + str(singlePrediction))