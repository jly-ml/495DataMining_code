#Jennifer Ly
#495
#Assignment 5


from sklearn import tree
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
import graphviz

dsQ1 = pd.read_csv('datasetQ1.csv')
dsQ1 = dsQ1.round(3)
x1 = []
x2 = []

for x in range(0,len(dsQ1.lagr) ):
    if dsQ1.lagr[x] != 0:
        print(dsQ1.lagr[x], '*', dsQ1.yi[x], '* (',dsQ1.xi1[x], ',', dsQ1.xi2[x],') +')
        x1_buf = dsQ1.lagr[x]* dsQ1.yi[x]* dsQ1.xi1[x]
        x2_buf = dsQ1.lagr[x]* dsQ1.yi[x]* dsQ1.xi2[x]
        x1.append(x1_buf)
        x2.append(x2_buf)

w = sum(x1)
x = sum(x2)
b = 1 - (w * dsQ1.xi1[0]  + x * dsQ1.xi2[0])

print('\n#1a. The hyperplane: h(x1,x2) =',w,'x1' ,'+',x,'x2', b,'\n')

def hyperplane_dist(x1,x2):
    dist = w*x1 + x*x2 + b
    if dist < -1:
        a = 'OUTSIDE the margin'
    else:
        a =  'INSIDE the margin'
    return dist,a
#b Distance of x6

print('#1b. Plugging in (1.9, 1.9) to h(x1,x2), the distance for X6 is:' , hyperplane_dist(dsQ1.xi2[5],dsQ1.xi2[5]))

print('\n#1c. Plugging in (3, 3) to h(x1,x2), the distance for Z is:' , hyperplane_dist(3,3), 'and will be classfied as 1\n')


print('--------------------------------------------------------------------------------------------------------')


dsQ2 = pd.read_csv('datasetQ2.csv')

z = {
    'A': 100,
    'B': 200
    }

dsQ2.Z = [z[x] for x in dsQ2.Z]
X = dsQ2.iloc[:,[0,1,2]]
Y = dsQ2.iloc[:,[3]]

X_train = dsQ2.iloc[0:8,0:3]
X_test = dsQ2.iloc[8:13,0:3]

y_test =  dsQ2.iloc[8:13,3]
y_train =  dsQ2.iloc[0:8,3]

clf_tree = DecisionTreeClassifier(criterion='gini')
clf_tree.fit(X_train, y_train)
a = ['X','Y','Z']
b = '12'
import graphviz
dot_data  = tree.export_graphviz(clf_tree, out_file = None,feature_names=a, class_names=b, filled = True, rounded= True)
decisiontree =  graphviz.Source(dot_data)
decisiontree.render('finaldecisiontree')






















