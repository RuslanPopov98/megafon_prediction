import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
get_ipython().run_line_magic('matplotlib', 'inline')

from graphviz  import Source
from IPython.display import SVG, display, HTML
style = "<style>svg{width: 70% !important; height: 60% !important;} </style>" 

contest_train = pd.read_csv(r'C:\Users\79776\Desktop\algothitm\megafon\contest_train.csv')
contest_test = pd.read_csv(r'C:\Users\79776\Desktop\algothitm\megafon\contest_test.csv')
sample_subm = pd.read_csv(r'C:\Users\79776\Desktop\algothitm\megafon\sample_subm.csv')
contest_test

X = contest_train.drop(['ID', 'TARGET'], axis=1)
y = contest_train.TARGET
X = X.replace(np.nan, 0).replace(np.inf, 0).replace(-np.inf, 0)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33, random_state=42)

#X.isnull().sum()
#X[~X.isin([np.nan, np.inf, -np.inf]).any(1)]
#X[~X.isin([np.nan, np.inf, -np.inf]).any(1)].astype(np.float64)
#X.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
#X[X.replace([np.inf, -np.inf], np.nan).notnull().all(axis=1)]
#ser = pd.Series(X['FEATURE_0'])
#ser[[0, 3, 6, 9]] 
#ser.head(50)
#X.head(50)

np.nan in X.values

XX=X.isnull().sum()

1 in XX.values

clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5)
clf.fit(X_train, y_train)
clf.score(X_train, y_train)

clf.score(X_test, y_test)

max_depth_value = range(1, 10)
score_data = pd.DataFrame()
for max_depth in max_depth_value:
    clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=max_depth)
    clf.fit(X_train, y_train)
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    
    mean_cross_val_score =  cross_val_score(clf, X_train, y_train, cv=5).mean()   
    temp_score_data = pd.DataFrame({'max_depth':[max_depth], 'train_score':[train_score], 'test_score':[test_score], 'cross_val_score':[mean_cross_val_score]})
    score_data = score_data.append(temp_score_data)

score_data_long = pd.melt(score_data, id_vars=['max_depth'], value_vars=['train_score','test_score', 'cross_val_score'], var_name='set_type', value_name='score')
sns.lineplot(x='max_depth', y='score', hue='set_type', data=score_data_long)

X_FinTest = contest_test.drop(['ID'], axis=1)
#y_FinTest = contest_train.TARGET
X_FinTest = X_FinTest.replace(np.nan, 0).replace(np.inf, 0).replace(-np.inf, 0)
#X_FinTest

predict = clf.predict(X_test)

from sklearn.metrics import f1_score
#predict = cat.predict(X_test)
f1_score(y_true=y_test, y_pred=predict, average="macro")
