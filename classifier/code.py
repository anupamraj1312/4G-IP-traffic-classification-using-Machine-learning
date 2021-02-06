import pandas as pd
import numpy as np
from sklearn import datasets
import sklearn
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics._classification import classification_report
from sklearn.metrics import precision_recall_fscore_support


# Graph Function
def plot_bar_x(label, class_training, title, xlabel, ylabel):
    index = np.arange(len(label))
    plt.bar(index, class_training, color=['red', 'blue', 'green', 'yellow', 'orange'])
    plt.xlabel(xlabel, fontsize=10)
    plt.ylabel(ylabel, fontsize=10)
    plt.xticks(index, label, fontsize=10, rotation=15)
    plt.title(title)
    plt.show()

    
# Preprocessing
video = pd.read_csv('IP data.csv')
size = video.shape
print(size)
target = video['label']
cols_to_drop = ['flow', 'src', 'dst', 'src_port', 'dst_port']
video_feature = video.drop(cols_to_drop, axis=1)
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
video_feature['label'] = label_encoder.fit_transform(video_feature['label'])
cols_to_drop = ['label']
target = label_encoder.fit_transform(target)
le_name_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print(le_name_mapping)
video_feature = video_feature.drop(cols_to_drop, axis=1)
scaler = StandardScaler();
df = scaler.fit_transform(video_feature)
df = pd.DataFrame(video_feature)
df = df.iloc[1:]
df = scaler.fit_transform(video_feature)
seed = 7  # To generate same sequence of random numbers

labels = ['Email', 'IM', 'P2P', 'VOIP', 'Web Media', 'WWW']

# Splitting the data for training and testing(80% train,20% test)
from sklearn.model_selection import train_test_split
train_data, test_data, train_label, test_label = train_test_split(df, target, test_size=0.25, random_state=seed)

# Naive Bayes
from sklearn.naive_bayes import GaussianNB
import time as t
classifier = GaussianNB()
t0 = t.time()
classifier = classifier.fit(train_data, train_label)
nbtt = round(t.time() - t0, 5)
print("training time nbc:", nbtt, "s")
t1 = t.time()
video_predicted_target = classifier.predict(test_data)
nbpt = round(t.time() - t1, 5)
print("predict time nbc :", nbpt, "s")
score1 = classifier.score(test_data, test_label)
print('Naive Bayes : ', score1)

# Random Forest
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_jobs=2, random_state=5)
t0 = t.time()
classifier = classifier.fit(train_data, train_label)
rftt = round(t.time() - t0, 5)
print("training time rfc:", rftt, "s")
t1 = t.time()
video_predicted_target = classifier.predict(test_data)
rfpt = round(t.time() - t1, 5)
print("predict time rfc:", rfpt, "s")
score2 = classifier.score(test_data, test_label)
print('Random Forest Classifier : ', score2)
y_pred = classifier.predict(test_data)
print(classification_report(test_label, y_pred))
rfscores = precision_recall_fscore_support(test_label, y_pred)
class_training = classifier.feature_importances_

# Decision Tree
from sklearn import tree
decision_tree = tree.DecisionTreeClassifier(criterion='gini')
classifier = decision_tree.fit(train_data, train_label)
print('The accuracy of the Decision Tree classifier on test data is {:.2f}'.format(decision_tree.score(test_data, test_label)))
t0 = t.time()
classifier = classifier.fit(train_data, train_label)
dttt = round(t.time() - t0, 5)
score3 = classifier.score(test_data, test_label)
print("training time of Decision tree :", dttt , "s")
t1 = t.time()
video_predicted_target = classifier.predict(test_data)
dtpt = round(t.time() - t1, 5)
print("predict time of decision tree:", dtpt , "s")
label = [ 'Source', 'Packet Size']
y_pred = classifier.predict(test_data)
print(classification_report(test_label, y_pred))
dtscores = precision_recall_fscore_support(test_label, y_pred)
#plot_bar_x(label, class_training, "Feature importance in Decision Tree", "Features", "Importance of feature")



#KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=7, p=2, metric='minkowski')
classifier = knn.fit(train_data, train_label)
print('The accuracy of the Knn classifier on test data is {:.2f}'.format(knn.score(test_data, test_label)))
t0 = t.time()
classifier = classifier.fit(train_data, train_label)
kntt = round(t.time() - t0, 5)
print("training time of knn :", kntt, "s")
t1 = t.time()
video_predicted_target = classifier.predict(test_data)
score4 = classifier.score(test_data, test_label)
knpt = round(t.time() - t1, 5)
print("predict time of knn:", knpt , "s")
 
# SVM
from sklearn.svm import SVC
svm = SVC(kernel='rbf', random_state=0, gamma=.10, C=1.0)
classifier = svm.fit(train_data, train_label)
print('The accuracy of the SVM classifier on test data is {:.2f}'.format(svm.score(test_data, test_label)))
t0 = t.time()
classifier = classifier.fit(train_data, train_label)
svtt = round(t.time() - t0, 5)
print("training time of SVM :", svtt, "s")
t1 = t.time()
video_predicted_target = classifier.predict(test_data)
svpt = round(t.time() - t1, 5)
score5 = classifier.score(test_data, test_label)
y_pred = classifier.predict(test_data)
svmscores = precision_recall_fscore_support(test_label, y_pred)
print("predict time of SVM :", svpt, "s")
 
label = ['Naive bayes', 'Random forest', 'decision tree', 'knn', 'svm']
 #Training Time graph
class_training = [
nbtt,
rftt,
dttt,
kntt,
svtt
]
plot_bar_x(label, class_training, "Training time", "Classifier", "time")
 
# Prediction time graph
class_prediction = [
nbpt,
rfpt,
dtpt,
knpt,
svpt
]
plot_bar_x(label, class_prediction, "Predict time", "classifiers", "time")
 
#Accuracy graph
class_score = [
score1 * 100,
score2 * 100,
score3 * 100,
score4 * 100,
score5 * 100
]
plot_bar_x(label, class_score, "Accuracy score", "classifiers", "percentage")

#Precision of Top 3 Algorithms
df = pd.DataFrame({'Random Forest': rfscores[0], 'Decision tree': dtscores[0], 'SVM': svmscores[0]}, labels)
ax = df.plot.bar(rot = 0, color = {'Random Forest':'green', 'Decision tree':'red', 'SVM':'blue'})
ax.set_title('Precision Scores')
ax.set_xlabel('Labels')
ax.set_ylabel('Precision')
plt.show()

#Recall of Top 3 Algorithms
df = pd.DataFrame({'Random Forest': rfscores[1], 'Decision tree': dtscores[1], 'SVM': svmscores[1]}, labels)
ax = df.plot.bar(rot = 0, color = {'Random Forest':'green', 'Decision tree':'red', 'SVM':'blue'})
ax.set_title('Recall Scores')
ax.set_xlabel('Labels')
ax.set_ylabel('Recall')
plt.show()

#F-scores of Top 3 Algorithms
df = pd.DataFrame({'Random Forest': rfscores[2], 'Decision tree': dtscores[2], 'SVM': svmscores[2]}, labels)
ax = df.plot.bar(rot = 0, color = {'Random Forest':'green', 'Decision tree':'red', 'SVM':'blue'})
ax.set_title('F-score Scores')
ax.set_xlabel('Labels')
ax.set_ylabel('F-score')
plt.show()













































