import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
df = pd.read_csv("Mall_Customers.csv")
df.drop("CustomerID", axis=1, inplace= True)

#Data Ingestion
print(df.head())
print(df.info())


#Plotting data into different ways
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Subplot 1: Spending score by gender
gender_spending = df.groupby("Gender")["Spending Score (1-100)"].sum().sort_values(ascending=False)
axes[0, 0].bar(gender_spending.index, gender_spending.values)
axes[0, 0].set_title("Total Spending Score by Gender")

# Subplot 2: Income and Spending Score by Gender
gender_group = df.groupby("Gender")[["Annual Income (k$)", "Spending Score (1-100)"]].sum()
gender_group.plot(kind='bar', ax=axes[0, 1])
axes[0, 1].set_title("Income and Spending Score by Gender")

# Subplot 3: Income and Spending Score by Age
age_group = df.groupby("Age")[["Annual Income (k$)", "Spending Score (1-100)"]].sum()
age_group.plot(ax=axes[1, 0])
axes[1, 0].set_title("Income and Spending Score by Age")

# Subplot 4: Scatter plot between Income and Spending Score
axes[1, 1].scatter(df["Annual Income (k$)"], df["Spending Score (1-100)"])
axes[1, 1].set_xlabel("Annual Income (k$)")
axes[1, 1].set_ylabel("Spending Score (1-100)")
axes[1, 1].set_title("Income vs Spending Score")

plt.tight_layout()


# Segregation of data into x and y
x = df.iloc[:, :-1]
y = df.iloc[:, 1]


# splitting of data into train and test data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# Scaling of data through Standardscaler and One Hot Encoder
cat_col = [col for col in df.columns if df[col].dtypes == "object"]
num_col = [col for col in df.columns if df[col].dtypes != "object"]
num_col.remove("Spending Score (1-100)")
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
num_pipeline = Pipeline([("SimpleImputer", SimpleImputer(strategy="median")),("StandardScaler", StandardScaler())])
cat_pipeline = Pipeline([("SimpleImputer", SimpleImputer(strategy="most_frequent")),("OneHotEncoder", OneHotEncoder())])
preprocessing = ColumnTransformer([("Numerical_Pipeline", num_pipeline, num_col),
                   ("Categorical_Pipleline", cat_pipeline, cat_col)])
x_train = preprocessing.fit_transform(x_train)
x_test = preprocessing.transform(x_test)


#Training of a LinearRegression Model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train, y_train)
pred = model.predict(x_test)


# Evaluating the model performance
from sklearn.metrics import r2_score
print(f"The r2 score of the linear model is: {r2_score(y_test, pred)}")









# Unsupervised Learning Practice
from sklearn.cluster import KMeans
# Training a kmeans clustering model
model = KMeans(n_clusters= 5)
pred = model.fit_predict(df[["Annual Income (k$)","Spending Score (1-100)"]])
plt.scatter(df["Annual Income (k$)"], df["Spending Score (1-100)"], c= pred)

# Evaluating the performance
from sklearn.metrics import silhouette_score
print("The silhouette score of your kmeans model is: ", silhouette_score(df[["Annual Income (k$)","Spending Score (1-100)"]], pred))

# Identifying the outliers/anamolies
from sklearn.ensemble import IsolationForest
detector = IsolationForest(contamination=0.2)
outliers = detector.fit_predict(df[["Annual Income (k$)","Spending Score (1-100)"]])
index = np.where(outliers<0)
x = df[["Annual Income (k$)","Spending Score (1-100)"]].values

#Plotting the Anamolies
plt.scatter(x=df["Annual Income (k$)"], y=df["Spending Score (1-100)"])
plt.scatter(x= x[index, 0], y=x[index, 1], edgecolors="red")
plt.show()
