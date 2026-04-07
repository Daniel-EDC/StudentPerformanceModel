import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas import read_csv
from sklearn.cluster import KMeans
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

df = pd.read_csv('Student_performance_Data _.csv')
df.head()

plt.scatter(df.GPA, df.Absences)
plt.xlabel("GPA")
plt.ylabel("Absences")
plt.show()

X = df[['Absences']]
y = df['GPA']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

km = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = km.fit_predict(X_scaled)

X_clustered = df[['Absences', 'Cluster']]

X_train, X_test, y_train, y_test = train_test_split(
    X_clustered,
    y,
    test_size=0.2,
    random_state=42
)

reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)

new_data = pd.DataFrame({'Absences': [5]})
new_data_scaled = scaler.transform(new_data)
new_cluster = km.predict(new_data_scaled)

new_data_clustered = pd.DataFrame({
    'Absences': [5],
    'Cluster': new_cluster
})

new_gpa_pred = reg.predict(new_data_clustered)
print("Predicted GPA:", new_gpa_pred[0])

score = reg.score(X_test, y_test)
print("Score:", score)

plt.scatter(y_test, y_pred)
plt.xlabel("Actual GPA")
plt.ylabel("Predicted GPA")
plt.title("Actual vs Predicted GPA")
plt.show()

def gpa_label(gpa):
    if gpa < 2.5:
        return 0
    elif gpa < 3.5:
        return 1
    else:
        return 2

df['GPA_Label'] = df['GPA'].apply(gpa_label)

cm = confusion_matrix(df['GPA_Label'], df['Cluster'])
print("Confusion Matrix:\n", cm)

print("\nClassification Report:")
print(classification_report(df['GPA_Label'], df['Cluster']))

for cluster in df['Cluster'].unique():
    cluster_data = df[df['Cluster'] == cluster]

    X_cluster = cluster_data[['Absences', 'Cluster']]
    y_cluster = cluster_data['GPA']

    sorted_idx = X_cluster['Absences'].argsort()
    X_sorted = X_cluster.iloc[sorted_idx]
    y_pred_cluster = reg.predict(X_sorted)

    plt.scatter(cluster_data['Absences'], y_cluster, label=f'Cluster {cluster}')
    plt.plot(X_sorted['Absences'], y_pred_cluster)

plt.xlabel("Absences")
plt.ylabel("GPA")
plt.show()
