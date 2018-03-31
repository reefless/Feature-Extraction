from sklearn import datasets, svm, neighbors, ensemble,model_selection, tree, multiclass
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.externals import joblib

#Read Data
train_data = genfromtxt('Tree.csv', delimiter=',', dtype='unicode')
test_data = genfromtxt('Test.csv', delimiter=',', dtype='unicode')

#Get class
x,x1,x2,x3,x4,y, z, class_data = train_data.T
class_data = class_data[1:]
print(test_data[1])

# delete second column 
train_data = np.delete(train_data,7,1)
test_data = np.delete(test_data,7,1)

# delete first row
train_data = np.delete(train_data,0,0)
test_data = np.delete(test_data,0,0)

'''class_data[class_data == 'Building'] = 0
class_data[class_data == 'Green Zone'] = 1
class_data[class_data == 'Canal'] = 2
class_data[class_data == 'Car'] = 3
class_data[class_data == 'Road'] = 4'''

#Training
#clf = neighbors.KNeighborsClassifier(n_neighbors=7)
clf = multiclass.OneVsRestClassifier(svm.SVC(kernel='linear'))
print(clf)
clf.fit(train_data, class_data)
print("Finish Training")
joblib.dump(clf, 'model.pkl')
#print(clf.predict([[6,120,130,135,130,1,4], [20,80,110,111,104,0,4], [13,94,95,90,92,0,5], [7,120,140,150,140,1,6], [70,40,50,45,46,0,3]]))
scores = model_selection.cross_val_score(clf, train_data, class_data)
print(scores.mean())
#print(clf.predict(test_data))
#print(len(train_data))
'''clf = neighbors.KNeighborsClassifier(15, weights='uniform')
clf.fit(train_data, class_data)
Z = clf.predict(test_data)
print(z)'''

