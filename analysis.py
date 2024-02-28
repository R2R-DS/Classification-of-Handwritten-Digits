# write your code here
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import GridSearchCV

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path="mnist.npz")

x_train = np.reshape(x_train, (60000, 784))

classes = np.unique(y_train)
min_val = x_train.min()
max_val = x_train.max()

X_train, X_test, Y_train, Y_test = train_test_split(x_train[0:6000], y_train[0:6000], test_size=0.3, random_state=40)

normalizer = Normalizer()
X_train_norm = normalizer.transform(X_train)
X_test_norm = normalizer.transform(X_test)

accuracy_dict = {}


def fit_predict_eval(model, features_train, features_test, target_train, target_test):
    # Fitting the model
    model.fit(features_train, target_train)
    # Making prediction
    prediction = model.predict(features_test)
    # Assessing accuracy and saving it to score
    score = accuracy_score(target_test, prediction)
    return round(score, 3)

KNN_param = {"n_neighbors": [3, 4],
             "weights": ["uniform", "distance"],
             "algorithm": ["auto", "brute"]}

RF_param = {"n_estimators": [300, 500],
            "max_features": ["sqrt", "log2"],
            "class_weight": ["balanced", "balanced_subsample"]}

KNN_grid_search = GridSearchCV(estimator=KNeighborsClassifier(),
                               param_grid=KNN_param,
                               scoring="accuracy",
                               n_jobs=-1)

RF_grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=40),
                              param_grid=RF_param,
                              scoring="accuracy",
                              n_jobs=-1)

KNN_grid_search.fit(X_train_norm, Y_train)
RF_grid_search.fit(X_train_norm, Y_train)

KNN_best_estimator = KNN_grid_search.best_estimator_
RF_best_estimator = RF_grid_search.best_estimator_

classifiers = [KNN_best_estimator,
               RF_best_estimator]

for classifier in classifiers:
    if classifier == KNN_best_estimator:
        print("K-nearest neighbours algorithm")
    elif classifier == RF_best_estimator:
        print("Random forest algorithm")
    print("best estimator:", classifier)
    print(f"""accuracy: {fit_predict_eval(model=classifier, 
                                          features_train=X_train_norm, 
                                          features_test=X_test_norm,
                                          target_train=Y_train,
                                          target_test=Y_test)}\n""")