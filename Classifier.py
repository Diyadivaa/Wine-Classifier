from sklearn.datasets import fetch_openml
X, y = fetch_openml('wine', version=1, return_X_y=True, as_frame=True)


print(X.columns)
print(y.unique())


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)


classifier.fit(X_train, y_train)


from sklearn.metrics import accuracy_score
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)
