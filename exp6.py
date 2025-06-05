from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report


iris = datasets.load_iris()
X = iris.data 
y = iris.target  


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


gnb = GaussianNB()


gnb.fit(X_train, y_train)


y_pred = gnb.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)


print("Accuracy:", accuracy)
print("Classification Report:\n", classification_rep)
