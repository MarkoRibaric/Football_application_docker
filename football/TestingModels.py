import pandas as pd
from cassandra.cluster import Cluster
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

cluster = Cluster(['127.0.0.1'])
session = cluster.connect('football_data')

rows = session.execute("SELECT * FROM football_data")
data = [(row.hshots, row.ashots, row.hstarget, row.astarget, row.hy, row.ay, row.hr, row.ar, 1 if row.ftr == 'H' else 0) for row in rows]
df = pd.DataFrame(data, columns=['hshots', 'ashots', 'hstarget', 'astarget', 'hy', 'ay', 'hr', 'ar', 'home_win'])

X = df[['hshots', 'ashots', 'hstarget', 'astarget', 'hy', 'ay', 'hr', 'ar']]
y = df['home_win']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Train set size:", X_train.shape[0])
print("Test set size:", X_test.shape[0])
# LOGISTIC REGRESSION
model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("LOGISTIC REGRESSION Accuracy:", accuracy)
print("LOGISTIC REGRESSION Classification Report:\n", report)


# DECISION TREE

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("DECISION TREE Accuracy:", accuracy)
print("DECISION TREE Classification Report:\n", report)

# RANDOM FOREST

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("RANDOM FOREST Accuracy:", accuracy)
print("RANDOM FOREST Classification Report:\n", report)

#SVM 

from sklearn.svm import SVC

model = SVC()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("SVM Accuracy:", accuracy)
print("SVM Classification Report:\n", report)

# GRADIENT BOOSTING

from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("GRADIENT BOOSTING Accuracy:", accuracy)
print("GRADIENT BOOSTING Classification Report:\n", report)

# NEURAL NETWORK

from sklearn.neural_network import MLPClassifier

model = MLPClassifier(max_iter=500)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("NEURAL NETWORK Accuracy:", accuracy)
print("NEURAL NETWORK Classification Report:\n", report)

session.shutdown()
cluster.shutdown()
