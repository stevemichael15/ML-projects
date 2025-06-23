import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
df = pd.read_csv("movie_genre_classification_final.csv")
df.drop(["Description", "Duration"], axis=1, inplace=True)

#transforming big data into smaller data
df["BoxOffice_USD_million"] = df["BoxOffice_USD"]/1000000
df["BoxOffice_USD_million"] = df["BoxOffice_USD_million"].astype(int)
df["Budget_USD_million"] = df["Budget_USD"]/1000000
df["Budget_USD_million"] = df["Budget_USD_million"].astype(int)


# creation of new features: performance and profit ----- Feature extraction
df["performance"] = "Hit"
for i in range(len(df["BoxOffice_USD_million"])):
    if df["BoxOffice_USD_million"][i]>=df["Budget_USD_million"][i]:
        df.loc[i, "performance"] = "Hit"
    elif df["BoxOffice_USD_million"][i]<df["Budget_USD_million"][i]:
        df.loc[i, "performance"] = "Flop"
df["profit"] = df["BoxOffice_USD_million"] - df["Budget_USD_million"]




# Visualizing data into different ways
# print(f"Lead actor and thier box Office Collection:\n {df.groupby('Lead_Actor')['BoxOffice_USD'].sum().sort_values(ascending=False).reset_index()}")
# print(f"Lead actor and the votes they have received:\n{df.groupby('Lead_Actor')['Votes'].sum().sort_values(ascending=False).reset_index()}")
# print(df[df["BoxOffice_USD"]== df["BoxOffice_USD"].max()][["Title", "Lead_Actor"]])
# print(df.groupby("Country")["BoxOffice_USD"].sum().sort_values(ascending= False).reset_index())
# print(df.groupby("Genre")["BoxOffice_USD"].sum().sort_values(ascending= False).reset_index())
# print(df.groupby("Director")["BoxOffice_USD"].sum().sort_values(ascending= False).reset_index())
# print(df.groupby("Lead_Actor")["Num_Awards"].sum().sort_values(ascending= False).reset_index())
# print(df["Lead_Actor"].value_counts().reset_index())# tells which lead actor has done how many movies
# print(df.groupby("Lead_Actor")["profit_billion"].sum())
# print(df.groupby("performance")["Critic_Reviews"].sum())


# Ploting data into different forms
# discrete_col = [col for col in df.columns if len(df[col].unique()) <=10]
# for col in discrete_col:
#     print(f"{col}: {df[col].unique()}")
# sns.boxplot(x=df["BoxOffice_USD"])
# print(df[["Critic_Reviews"]])
# df.groupby(["Lead_Actor", "performance"]).size().unstack(fill_value=0).plot(kind="bar")
# plt.xticks(rotation=30)
# print(df.groupby("performance")["Rating"].sum())
# print(df.groupby("performance")["Num_Awards"].sum())
# print(df.info())
# print(df)
# df.groupby("Lead_Actor")["BoxOffice_USD"].sum().sort_values(ascending=False).reset_index().plot(kind= "bar", color="red")
# df.groupby("Lead_Actor")["profit_billion"].sum().sort_values(ascending=False).reset_index().plot(kind="bar")
# plt.xlabel("Lead Actor")
# plt.ylabel("Profit")
# plt.grid(True)
# plt.show()





# dropping unwanted and meaningless features------- feature selection
df.drop(["BoxOffice_USD","Budget_USD", "BoxOffice_USD_million", "Budget_USD_million", "Genre", "Director", "Lead_Actor", "Language", "Production_Company", "Country", "Year", "Title", "Content_Rating"], axis= 1, inplace= True)
df["performance"] = df["performance"].replace(to_replace={"Hit":1, "Flop":0})





# Dividing data into independent variables and target features
x = df.drop("performance", axis= 1)
y = df["performance"]


#Features selection through VIF, RFE and PCA
# VIF
from statsmodels.stats.outliers_influence import  variance_inflation_factor
vif = pd.DataFrame()
vif["Features"] = df.columns
vif["vif"] = [variance_inflation_factor(df.values, i)for i in range(len(df.columns))]
print(vif)


# RFE
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
rfe = RFE(LogisticRegression(), n_features_to_select=3)
rfe.fit(x, y)
print("RFE: ")
print(f"Important features based on true or false: {rfe.support_}") #selects the features and give true and false
print(f"Important features based on their ranking: {rfe.ranking_}") #shows the importance of a feature ranging from 1








#Splitting data into train and test data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.3, random_state=1)
print(x_train)


#Scaling the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)




#making a Logistic ML model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train, y_train)
coef = model.coef_
y_pred = model.predict(x_test)


# checking the model accuracy with the test data
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
print(f"Confusion Matrix: \n {confusion_matrix(y_test, y_pred)}")
print(f"Classification Report: {classification_report(y_test, y_pred)}")
print(f"Accuracy Score: {accuracy_score(y_test, y_pred)}")


cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

