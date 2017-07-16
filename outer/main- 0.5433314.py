
# coding: utf-8

# In[24]:

import pandas as pd
import numpy as np
from gensim.models import word2vec
# отключим всякие предупреждения Anaconda
import warnings
warnings.filterwarnings('ignore')
get_ipython().magic('matplotlib inline')
from matplotlib import pyplot as plt
import seaborn as sns

import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import SGDClassifier
from scipy.sparse import csr_matrix
from sklearn.linear_model import Ridge
from scipy.sparse import hstack
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.grid_search import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import mean_squared_error
import re
from nltk.corpus import stopwords
import pymorphy2


# In[25]:

test = pd.read_csv('ml5/test.csv',sep = ';',na_values = 'None')
train = pd.read_csv('ml5/train.csv',sep = ';',na_values = 'None')
y = train.pop('cardio')


# In[26]:

train.head()


# In[27]:

test.info()


# In[29]:

test.head()


# In[30]:

import pandas as pd;
import random
import numpy as np;
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
get_ipython().magic('matplotlib inline')


# In[31]:

dtrain = xgb.DMatrix(train, label=y)
dtest = xgb.DMatrix(test)


# In[38]:

get_ipython().run_cell_magic('time', '', "param = {'max_depth':4, 'eta':0.01, 'silent':0,\n         'objective':'binary:logistic',\n         'subsample':0.8,\n         'colsample_bytree':0.8,\n         'seed':202,\n         #'scale_pos_weight':scale_pos_weight   с этой штуковиной почему-то результат был на мнооого хуже\n         'updater':'grow_gpu'\n        }\nparam['eval_metric'] = 'logloss'\ntrees = 5000\ncv = xgb.cv(param, dtrain, metrics=('logloss'), show_stdv=True,\n            num_boost_round=trees,nfold=5,early_stopping_rounds = 50)")


# In[40]:

cv[1000:].plot(y=['test-logloss-mean', 'train-logloss-mean'], secondary_y='train-logloss-mean')
print cv.loc[cv['test-logloss-mean'].argmin()]
trees = cv['test-logloss-mean'].argmin()


# In[41]:

get_ipython().run_cell_magic('time', '', 'bst = xgb.train(param,dtrain,trees)\na = pd.DataFrame()')


# In[42]:

a['y'] = bst.predict(dtest)
a.to_csv('xgb.csv', index = False, header = False)


# In[ ]:



