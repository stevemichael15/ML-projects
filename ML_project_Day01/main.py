import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
df = sns.load_dataset("iris")
print(df.head())
print(df.info())

# Label encoding performed manually
df["species"] = df["species"].replace(to_replace={"setosa": 0, "versicolor": 1, "virginica": 2})


#-------------------------------visualizing species based on their physical charateristics
# plt.figure(figsize=(10, 8))
# plt.subplot(1, 2, 1)
# plt.scatter(x= df["sepal_length"], y= df["sepal_width"], c=df["species"])
# plt.xlabel("Sepal_length")
# plt.ylabel("Sepal_Width")
# plt.subplot(1, 2, 2)
# plt.scatter(x= df["petal_length"], y= df["petal_width"], c=df["species"])
# plt.xlabel("Petal_length")
# plt.ylabel("Petal_Width")
# plt.show()


#division of data into Independent variables and target feature
x, y = df.iloc[:, :-1], df.iloc[:, -1]

#dividing the data further into train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

#scaling the data(optional)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# training a Logistic Model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train, y_train)
coef = model.coef_
y_pred = model.predict(x_test)

# Evaluating the performance
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
accuracyScore = accuracy_score(y_test, y_pred)
classificationReport = classification_report(y_test, y_pred)
confusionMatrix = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix: {confusionMatrix}")
print(f"Classification Report: {classificationReport}")
print(f"Accuracy Score: {accuracyScore}")
