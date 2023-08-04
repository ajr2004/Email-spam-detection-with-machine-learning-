import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Load dataset
df = pd.read_csv("spam.csv", encoding="latin1")

# Check column list present in df
print("Columns present in the dataset:")
print(df.columns)

# Check descriptive statistics
print("\nDescriptive statistics of the dataset:")
print(df.info())

# Check the number of rows and columns present in df
print("\nNumber of rows: ", df.shape[0])
print("Number of columns: ", df.shape[1])

# Let's see null value count in df
print("\nNull value count in the dataset:")
print(df.isnull().sum())

# Drop unnecessary columns
df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)

# Rename columns names for easy understanding
df.columns = ['spam/ham', 'sms']

# Convert the text data into numerical form
df.loc[df['spam/ham'] == 'spam', 'spam/ham'] = 0
df.loc[df['spam/ham'] == 'ham', 'spam/ham'] = 1

# Devide x and y parameters to train model
x = df.sms
y = df['spam/ham']

# Devide the whole dataset into training and testing set for model training
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=3)

# Convert text data into numerical vectors using TfidfVectorizer
feat_vect = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)

ytrain = ytrain.astype('int')
ytest = ytest.astype('int')

xtrain_vec = feat_vect.fit_transform(xtrain)
xtest_vec = feat_vect.transform(xtest)

# Initialize and train the logistic regression model
logi = LogisticRegression()
logi.fit(xtrain_vec, ytrain)

# Evaluate the model on training and testing data
train_accuracy = logi.score(xtrain_vec, ytrain)
test_accuracy = logi.score(xtest_vec, ytest)
print("\nTraining Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)

# Make predictions on the test data
pred_logi = logi.predict(xtest_vec)

# Evaluate the model using various metrics
print("\nAccuracy Score:", accuracy_score(ytest, pred_logi))
print("Confusion Matrix:")
print(confusion_matrix(ytest, pred_logi))
print("Classification Report:")
print(classification_report(ytest, pred_logi))
