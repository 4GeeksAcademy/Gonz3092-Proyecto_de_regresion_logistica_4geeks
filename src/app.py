import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.linear_model import LogisticRegression

url = "https://raw.githubusercontent.com/4GeeksAcademy/logistic-regression-project-tutorial/main/bank-marketing-campaign-data.csv"
data = pd.read_csv(url, sep=";")

data = data.drop_duplicates().reset_index(drop = True)

data["job"] = pd.factorize(data["job"])[0]
data["marital"] = pd.factorize(data["marital"])[0]
data["education"] = pd.factorize(data["education"])[0]
data["default"] = pd.factorize(data["default"])[0]
data["housing"] = pd.factorize(data["housing"])[0]
data["loan"] = pd.factorize(data["loan"])[0]
data["contact"] = pd.factorize(data["contact"])[0]
data["month"] = pd.factorize(data["month"])[0]
data["day_of_week"] = pd.factorize(data["day_of_week"])[0]
data["poutcome"] = pd.factorize(data["poutcome"])[0]
data["y"] = pd.factorize(data["y"])[0]

num_variables = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "day_of_week", "poutcome", "age", "duration", 
                 "campaign", "pdays", "previous", "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]

X = data.drop("y", axis = 1)
y = data["y"]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()

x_train_scaled = pd.DataFrame(scaler.fit_transform(x_train), columns=num_variables)
x_test_scaled = pd.DataFrame(scaler.transform(x_test), columns=num_variables)

selection_model = SelectKBest(f_classif, k = 5)
selection_model.fit(x_train_scaled, y_train)
ix = selection_model.get_support()
x_train_sel = pd.DataFrame(selection_model.transform(x_train_scaled), columns = x_train_scaled.columns.values[ix])
x_test_sel = pd.DataFrame(selection_model.transform(x_test_scaled), columns = x_test_scaled.columns.values[ix])

x_train_sel["y"] = list(y_train)
x_test_sel["y"] = list(y_test)

x_train_sel.to_csv("../data/processed/clean_train.csv", index = False)
x_test_sel.to_csv("../data/processed/clean_test.csv", index = False)

train_data = pd.read_csv("../data/processed/clean_train.csv")
test_data = pd.read_csv("../data/processed/clean_test.csv")

x_train = train_data.drop(["y"], axis = 1)
y_train = train_data["y"]

x_test = test_data.drop(["y"], axis = 1)
y_test = test_data["y"]

model = LogisticRegression(max_iter=2000)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
