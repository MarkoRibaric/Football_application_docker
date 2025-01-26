import pandas as pd
import pickle
from cassandra.cluster import Cluster
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import json
import dill

cluster = Cluster(['127.0.0.1'])
session = cluster.connect('football_data')

query = """
    SELECT hshots, ashots, hstarget, astarget, hy, ay, hr, ar, ftr
    FROM football_data
"""
rows = session.execute(query)

# Convert the query result into a list of tuples
data = [row for row in rows]

# Create a pandas DataFrame from the fetched data
df = pd.DataFrame(data, columns=['hshots', 'ashots', 'hstarget', 'astarget', 'hy', 'ay', 'hr', 'ar', 'ftr'])

df['result'] = df['ftr'].map({'H': 1, 'A': 2, 'D': 0})

X = df[['hshots', 'ashots', 'hstarget', 'astarget', 'hy', 'ay', 'hr', 'ar']]
y = df['result']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Train set size:", X_train.shape[0])
print("Test set size:", X_test.shape[0])

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=['Draw', 'Home Win', 'Away Win']))

model_filename = 'app/random_forest_model.pkl'
with open(model_filename, 'wb') as file:
    dill.dump(model, file)

print(f"Model saved to {model_filename}")
