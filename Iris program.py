import numpy as np
import pandas as pd
from sklearn import neighbors
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree

df = pd.read_csv('iris.csv')

print(df.head(10),  '\n')

print(df.shape,  '\n')

print(df.columns,  '\n')

print(df['variety'].value_counts(),  '\n')


#  sepal.length ,  sepal.width plot
df.plot(kind='scatter',  x='sepal.length',  y='sepal.width')
plt.show()

df.plot(kind='scatter',  x='petal.length',  y='petal.width')
plt.show()


# petal.length,  petal.width plot
sns.set_style('whitegrid');
sns.FacetGrid(df,  hue='variety',  height=4)\
    .map(plt.scatter,  'sepal.length',  'sepal.width')\
    .add_legend();
plt.show()

sns.barplot(x='petal.length', y='petal.width', data=df)
plt.show()

#  define X and y
X = np.array(df.drop(['variety'],  axis=1))
y = df['variety']


df.hist()
plt.show()

#  Train Test model
X_train,  X_test,  y_train,  y_test = train_test_split(X,  y,  test_size=.2)


#  KNearest classifier
clf = neighbors.KNeighborsClassifier(n_neighbors=3)

#  find patterns in data
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print('KNN : ', accuracy)


#  decision tree classifier
clf = tree.DecisionTreeClassifier()
clf.fit(X_train,  y_train)

accuracy = clf.score(X_test,  y_test)
print('Decision Tree : ', accuracy)

print(X[110], y[110])

example_measure=np.array([[5.9,  3.2,  3.5,  .8]])

example_measure=example_measure.reshape(len(example_measure), -1)
prediction=clf.predict(example_measure)
print(prediction)


