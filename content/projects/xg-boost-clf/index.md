---
title: XGBoost Classifier for Music Recommendation
date: 2023-07-28T23:00:00-04:00
draft: false
ShowToc: true
math: true
tags: 
    - xg-boost
    - music-recommender
    - recommender-system
    - music-classifier
cover:
    image: images/recommender-system.jpg
    relative: true # To use relative path for cover image, used in hugo Page-bundles
---

# Section 0 Preface for Imports, Data Handling, & Methodologies 

## Section 0.1 Preface for Write-Up Interpretation & Acknowledgements

Code was used from my own Github repository, found at `www.github.com/sunnydigital/IDS_F21`, including code derived from Stephen Spivak from Introduction to Data Science, Fall 2021. Most of the code falling under the aforementioned two categories surrounds the `PCA` and `k-means` analysis plots.

## Section 0.2 Imports & Installation of Packages, Libraries, Seaborn Settings, and Dataset(s)

Below we set the random seed to the numeric portion of my ID: `N12345678` import packages & libraries as well as set the settings for `seaborn` plots


```python
!pip install xgboost
!pip install impyute
!pip install missingno
!pip install eli5
!pip install scikit-optimize
!pip install tune-sklearn ray[tune]
!pip install scikit-plot
!pip install colorcet==3.0.0
!pip install yellowbrick
```

    Requirement already satisfied: xgboost in /Users/sunnyson/opt/anaconda3/lib/python3.9/site-packages (1.7.3)
    Requirement already satisfied: numpy in /Users/sunnyson/opt/anaconda3/lib/python3.9/site-packages (from xgboost) (1.21.5)
    Requirement already satisfied: scipy in /Users/sunnyson/opt/anaconda3/lib/python3.9/site-packages (from xgboost) (1.7.3)
    Requirement already satisfied: impyute in /Users/sunnyson/opt/anaconda3/lib/python3.9/site-packages (0.0.8)
    Requirement already satisfied: scipy in /Users/sunnyson/opt/anaconda3/lib/python3.9/site-packages (from impyute) (1.7.3)
    Requirement already satisfied: scikit-learn in /Users/sunnyson/opt/anaconda3/lib/python3.9/site-packages (from impyute) (1.0.2)
    Requirement already satisfied: numpy in /Users/sunnyson/opt/anaconda3/lib/python3.9/site-packages (from impyute) (1.21.5)
    Requirement already satisfied: threadpoolctl>=2.0.0 in /Users/sunnyson/opt/anaconda3/lib/python3.9/site-packages (from scikit-learn->impyute) (2.2.0)
    Requirement already satisfied: joblib>=0.11 in /Users/sunnyson/opt/anaconda3/lib/python3.9/site-packages (from scikit-learn->impyute) (1.1.0)
    Requirement already satisfied: missingno in /Users/sunnyson/opt/anaconda3/lib/python3.9/site-packages (0.5.1)
    Requirement already satisfied: numpy in /Users/sunnyson/opt/anaconda3/lib/python3.9/site-packages (from missingno) (1.21.5)
    Requirement already satisfied: seaborn in /Users/sunnyson/opt/anaconda3/lib/python3.9/site-packages (from missingno) (0.11.2)
    Requirement already satisfied: scipy in /Users/sunnyson/opt/anaconda3/lib/python3.9/site-packages (from missingno) (1.7.3)
    Requirement already satisfied: matplotlib in /Users/sunnyson/opt/anaconda3/lib/python3.9/site-packages (from missingno) (3.5.1)
    Requirement already satisfied: kiwisolver>=1.0.1 in /Users/sunnyson/opt/anaconda3/lib/python3.9/site-packages (from matplotlib->missingno) (1.4.2)
    Requirement already satisfied: packaging>=20.0 in /Users/sunnyson/opt/anaconda3/lib/python3.9/site-packages (from matplotlib->missingno) (21.3)
    Requirement already satisfied: pyparsing>=2.2.1 in /Users/sunnyson/opt/anaconda3/lib/python3.9/site-packages (from matplotlib->missingno) (3.0.4)
    Requirement already satisfied: cycler>=0.10 in /Users/sunnyson/opt/anaconda3/lib/python3.9/site-packages (from matplotlib->missingno) (0.11.0)
    Requirement already satisfied: pillow>=6.2.0 in /Users/sunnyson/opt/anaconda3/lib/python3.9/site-packages (from matplotlib->missingno) (9.0.1)
    Requirement already satisfied: fonttools>=4.22.0 in /Users/sunnyson/opt/anaconda3/lib/python3.9/site-packages (from matplotlib->missingno) (4.25.0)
    Requirement already satisfied: python-dateutil>=2.7 in /Users/sunnyson/opt/anaconda3/lib/python3.9/site-packages (from matplotlib->missingno) (2.8.2)
    Requirement already satisfied: six>=1.5 in /Users/sunnyson/opt/anaconda3/lib/python3.9/site-packages (from python-dateutil>=2.7->matplotlib->missingno) (1.16.0)
    Requirement already satisfied: pandas>=0.23 in /Users/sunnyson/opt/anaconda3/lib/python3.9/site-packages (from seaborn->missingno) (1.4.2)
    Requirement already satisfied: pytz>=2020.1 in /Users/sunnyson/opt/anaconda3/lib/python3.9/site-packages (from pandas>=0.23->seaborn->missingno) (2021.3)
    Requirement already satisfied: eli5 in /Users/sunnyson/opt/anaconda3/lib/python3.9/site-packages (0.13.0)
    Requirement already satisfied: six in /Users/sunnyson/opt/anaconda3/lib/python3.9/site-packages (from eli5) (1.16.0)
    Requirement already satisfied: numpy>=1.9.0 in /Users/sunnyson/opt/anaconda3/lib/python3.9/site-packages (from eli5) (1.21.5)
    Requirement already satisfied: attrs>17.1.0 in /Users/sunnyson/opt/anaconda3/lib/python3.9/site-packages (from eli5) (21.4.0)
    Requirement already satisfied: scikit-learn>=0.20 in /Users/sunnyson/opt/anaconda3/lib/python3.9/site-packages (from eli5) (1.0.2)
    Requirement already satisfied: jinja2>=3.0.0 in /Users/sunnyson/opt/anaconda3/lib/python3.9/site-packages (from eli5) (3.0.3)
    Requirement already satisfied: scipy in /Users/sunnyson/opt/anaconda3/lib/python3.9/site-packages (from eli5) (1.7.3)
    Requirement already satisfied: graphviz in /Users/sunnyson/opt/anaconda3/lib/python3.9/site-packages (from eli5) (0.20.1)
    Requirement already satisfied: tabulate>=0.7.7 in /Users/sunnyson/opt/anaconda3/lib/python3.9/site-packages (from eli5) (0.8.9)
    Requirement already satisfied: MarkupSafe>=2.0 in /Users/sunnyson/opt/anaconda3/lib/python3.9/site-packages (from jinja2>=3.0.0->eli5) (2.0.1)
    Requirement already satisfied: threadpoolctl>=2.0.0 in /Users/sunnyson/opt/anaconda3/lib/python3.9/site-packages (from scikit-learn>=0.20->eli5) (2.2.0)
    Requirement already satisfied: joblib>=0.11 in /Users/sunnyson/opt/anaconda3/lib/python3.9/site-packages (from scikit-learn>=0.20->eli5) (1.1.0)
    Requirement already satisfied: scikit-optimize in /Users/sunnyson/opt/anaconda3/lib/python3.9/site-packages (0.9.0)
    Requirement already satisfied: scikit-learn>=0.20.0 in /Users/sunnyson/opt/anaconda3/lib/python3.9/site-packages (from scikit-optimize) (1.0.2)
    Requirement already satisfied: pyaml>=16.9 in /Users/sunnyson/opt/anaconda3/lib/python3.9/site-packages (from scikit-optimize) (21.10.1)
    Requirement already satisfied: numpy>=1.13.3 in /Users/sunnyson/opt/anaconda3/lib/python3.9/site-packages (from scikit-optimize) (1.21.5)
    Requirement already satisfied: joblib>=0.11 in /Users/sunnyson/opt/anaconda3/lib/python3.9/site-packages (from scikit-optimize) (1.1.0)
    Requirement already satisfied: scipy>=0.19.1 in /Users/sunnyson/opt/anaconda3/lib/python3.9/site-packages (from scikit-optimize) (1.7.3)
    Requirement already satisfied: PyYAML in /Users/sunnyson/opt/anaconda3/lib/python3.9/site-packages (from pyaml>=16.9->scikit-optimize) (6.0)
    Requirement already satisfied: threadpoolctl>=2.0.0 in /Users/sunnyson/opt/anaconda3/lib/python3.9/site-packages (from scikit-learn>=0.20.0->scikit-optimize) (2.2.0)
    zsh:1: no matches found: ray[tune]
    Requirement already satisfied: scikit-plot in /Users/sunnyson/opt/anaconda3/lib/python3.9/site-packages (0.3.7)
    Requirement already satisfied: scikit-learn>=0.18 in /Users/sunnyson/opt/anaconda3/lib/python3.9/site-packages (from scikit-plot) (1.0.2)
    Requirement already satisfied: matplotlib>=1.4.0 in /Users/sunnyson/opt/anaconda3/lib/python3.9/site-packages (from scikit-plot) (3.5.1)
    Requirement already satisfied: scipy>=0.9 in /Users/sunnyson/opt/anaconda3/lib/python3.9/site-packages (from scikit-plot) (1.7.3)
    Requirement already satisfied: joblib>=0.10 in /Users/sunnyson/opt/anaconda3/lib/python3.9/site-packages (from scikit-plot) (1.1.0)
    Requirement already satisfied: cycler>=0.10 in /Users/sunnyson/opt/anaconda3/lib/python3.9/site-packages (from matplotlib>=1.4.0->scikit-plot) (0.11.0)
    Requirement already satisfied: packaging>=20.0 in /Users/sunnyson/opt/anaconda3/lib/python3.9/site-packages (from matplotlib>=1.4.0->scikit-plot) (21.3)
    Requirement already satisfied: python-dateutil>=2.7 in /Users/sunnyson/opt/anaconda3/lib/python3.9/site-packages (from matplotlib>=1.4.0->scikit-plot) (2.8.2)
    Requirement already satisfied: pyparsing>=2.2.1 in /Users/sunnyson/opt/anaconda3/lib/python3.9/site-packages (from matplotlib>=1.4.0->scikit-plot) (3.0.4)
    Requirement already satisfied: pillow>=6.2.0 in /Users/sunnyson/opt/anaconda3/lib/python3.9/site-packages (from matplotlib>=1.4.0->scikit-plot) (9.0.1)
    Requirement already satisfied: numpy>=1.17 in /Users/sunnyson/opt/anaconda3/lib/python3.9/site-packages (from matplotlib>=1.4.0->scikit-plot) (1.21.5)
    Requirement already satisfied: fonttools>=4.22.0 in /Users/sunnyson/opt/anaconda3/lib/python3.9/site-packages (from matplotlib>=1.4.0->scikit-plot) (4.25.0)
    Requirement already satisfied: kiwisolver>=1.0.1 in /Users/sunnyson/opt/anaconda3/lib/python3.9/site-packages (from matplotlib>=1.4.0->scikit-plot) (1.4.2)
    Requirement already satisfied: six>=1.5 in /Users/sunnyson/opt/anaconda3/lib/python3.9/site-packages (from python-dateutil>=2.7->matplotlib>=1.4.0->scikit-plot) (1.16.0)
    Requirement already satisfied: threadpoolctl>=2.0.0 in /Users/sunnyson/opt/anaconda3/lib/python3.9/site-packages (from scikit-learn>=0.18->scikit-plot) (2.2.0)
    Requirement already satisfied: colorcet==3.0.0 in /Users/sunnyson/opt/anaconda3/lib/python3.9/site-packages (3.0.0)
    Requirement already satisfied: param>=1.7.0 in /Users/sunnyson/opt/anaconda3/lib/python3.9/site-packages (from colorcet==3.0.0) (1.12.0)
    Requirement already satisfied: pyct>=0.4.4 in /Users/sunnyson/opt/anaconda3/lib/python3.9/site-packages (from colorcet==3.0.0) (0.4.6)
    Requirement already satisfied: yellowbrick in /Users/sunnyson/opt/anaconda3/lib/python3.9/site-packages (1.5)

    Requirement already satisfied: scikit-learn>=1.0.0 in /Users/sunnyson/opt/anaconda3/lib/python3.9/site-packages (from yellowbrick) (1.0.2)

    Requirement already satisfied: numpy>=1.16.0 in /Users/sunnyson/opt/anaconda3/lib/python3.9/site-packages (from yellowbrick) (1.21.5)

    Requirement already satisfied: matplotlib!=3.0.0,>=2.0.2 in /Users/sunnyson/opt/anaconda3/lib/python3.9/site-packages (from yellowbrick) (3.5.1)

    Requirement already satisfied: cycler>=0.10.0 in /Users/sunnyson/opt/anaconda3/lib/python3.9/site-packages (from yellowbrick) (0.11.0)

    Requirement already satisfied: scipy>=1.0.0 in /Users/sunnyson/opt/anaconda3/lib/python3.9/site-packages (from yellowbrick) (1.7.3)

    Requirement already satisfied: packaging>=20.0 in /Users/sunnyson/opt/anaconda3/lib/python3.9/site-packages (from matplotlib!=3.0.0,>=2.0.2->yellowbrick) (21.3)

    Requirement already satisfied: fonttools>=4.22.0 in /Users/sunnyson/opt/anaconda3/lib/python3.9/site-packages (from matplotlib!=3.0.0,>=2.0.2->yellowbrick) (4.25.0)

    Requirement already satisfied: kiwisolver>=1.0.1 in /Users/sunnyson/opt/anaconda3/lib/python3.9/site-packages (from matplotlib!=3.0.0,>=2.0.2->yellowbrick) (1.4.2)

    Requirement already satisfied: python-dateutil>=2.7 in /Users/sunnyson/opt/anaconda3/lib/python3.9/site-packages (from matplotlib!=3.0.0,>=2.0.2->yellowbrick) (2.8.2)

    Requirement already satisfied: pillow>=6.2.0 in /Users/sunnyson/opt/anaconda3/lib/python3.9/site-packages (from matplotlib!=3.0.0,>=2.0.2->yellowbrick) (9.0.1)

    Requirement already satisfied: pyparsing>=2.2.1 in /Users/sunnyson/opt/anaconda3/lib/python3.9/site-packages (from matplotlib!=3.0.0,>=2.0.2->yellowbrick) (3.0.4)

    Requirement already satisfied: six>=1.5 in /Users/sunnyson/opt/anaconda3/lib/python3.9/site-packages (from python-dateutil>=2.7->matplotlib!=3.0.0,>=2.0.2->yellowbrick) (1.16.0)

    Requirement already satisfied: joblib>=0.11 in /Users/sunnyson/opt/anaconda3/lib/python3.9/site-packages (from scikit-learn>=1.0.0->yellowbrick) (1.1.0)

    Requirement already satisfied: threadpoolctl>=2.0.0 in /Users/sunnyson/opt/anaconda3/lib/python3.9/site-packages (from scikit-learn>=1.0.0->yellowbrick) (2.2.0)



Below we set the random seed to the numerical portion of our NYU ID (as per the spec sheet).


```python
import random
random.seed(12345678)
```


```python
import sys
import os
import time
import re

import numpy as np
import pandas as pd

from statsmodels.distributions.empirical_distribution import ECDF
import statsmodels.api as sm

import scipy.stats as st
import statsmodels.api as sm ## Need revision for Windows use
from scipy.stats import zscore
from scipy.spatial.distance import squareform

from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression, Perceptron
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_validate, KFold, cross_val_score, RepeatedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, make_scorer, confusion_matrix, plot_confusion_matrix, mean_squared_error as mse, f1_score
from sklearn import tree, ensemble, metrics, calibration
from sklearn.svm import SVC, LinearSVC
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_samples
from sklearn.impute import KNNImputer
from sklearn.ensemble import AdaBoostClassifier

from IPython import display

from eli5.sklearn import PermutationImportance
from eli5 import show_prediction
import eli5

from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_objective
from skopt import BayesSearchCV

'''from tune_sklearn import TuneSearchCV, TuneGridSearchCV
import ray.tune as tune'''

from tqdm import tqdm

from matplotlib import pyplot as plt
from matplotlib import mlab
from matplotlib.pyplot import figure
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
import scikitplot as skplt
import seaborn as sns; sns.set_theme(color_codes=True); sns.set_style("whitegrid")
import graphviz
import colorcet as cc

from impyute.imputation.cs import fast_knn

from xgboost.sklearn import XGBClassifier
import xgboost as xgb

from yellowbrick.classifier import ROCAUC
```

## Section 0.3 `NaN` Data Visual Inspection

Below we view the potentially missing data out of the dataset we obtained. Missing data (yellow) are greatly contrasted with the present data (blue).

This was done to visually inspect the proportion of data missing versus those present.


```python
data = 'musicData.csv'
df = pd.read_csv(data)

colors = ['#000099', '#ffff00']
fig, ax = plt.subplots(figsize=(100,10))

sns.heatmap(df.isnull(), cmap=sns.color_palette(colors), ax=ax)
```




    <AxesSubplot:>




    
![png](images/Sunny%20Son%20IML%20Capstone_6_1.png)
    


## Section 0.4 `string` Data Removal

Through our goal of classification/unsupervised learning, we will not include `string` data, as in the scope of this analysis, we will not be considering tthe potentially lingusitic properties of `artist_name`, `track_name`, `obtained_date`. Furthermore, `instance_id` is simply the unique classifier of an individual song, and given we are not considering the properties associated with a song, we will not need the specific label.


```python
df_map = df[['instance_id', 'artist_name', 'track_name', 'obtained_date']]
df = df.loc[:, ~df.columns.isin(['instance_id', 'artist_name','track_name','obtained_date'])]
```


```python
df.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>popularity</th>
      <th>acousticness</th>
      <th>danceability</th>
      <th>duration_ms</th>
      <th>energy</th>
      <th>instrumentalness</th>
      <th>key</th>
      <th>liveness</th>
      <th>loudness</th>
      <th>mode</th>
      <th>speechiness</th>
      <th>tempo</th>
      <th>valence</th>
      <th>music_genre</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>27.0</td>
      <td>0.00468</td>
      <td>0.652</td>
      <td>-1.0</td>
      <td>0.941</td>
      <td>0.792000</td>
      <td>A#</td>
      <td>0.1150</td>
      <td>-5.201</td>
      <td>Minor</td>
      <td>0.0748</td>
      <td>100.889</td>
      <td>0.759</td>
      <td>Electronic</td>
    </tr>
    <tr>
      <th>1</th>
      <td>31.0</td>
      <td>0.01270</td>
      <td>0.622</td>
      <td>218293.0</td>
      <td>0.890</td>
      <td>0.950000</td>
      <td>D</td>
      <td>0.1240</td>
      <td>-7.043</td>
      <td>Minor</td>
      <td>0.0300</td>
      <td>115.00200000000001</td>
      <td>0.531</td>
      <td>Electronic</td>
    </tr>
    <tr>
      <th>2</th>
      <td>28.0</td>
      <td>0.00306</td>
      <td>0.620</td>
      <td>215613.0</td>
      <td>0.755</td>
      <td>0.011800</td>
      <td>G#</td>
      <td>0.5340</td>
      <td>-4.617</td>
      <td>Major</td>
      <td>0.0345</td>
      <td>127.994</td>
      <td>0.333</td>
      <td>Electronic</td>
    </tr>
    <tr>
      <th>3</th>
      <td>34.0</td>
      <td>0.02540</td>
      <td>0.774</td>
      <td>166875.0</td>
      <td>0.700</td>
      <td>0.002530</td>
      <td>C#</td>
      <td>0.1570</td>
      <td>-4.498</td>
      <td>Major</td>
      <td>0.2390</td>
      <td>128.014</td>
      <td>0.270</td>
      <td>Electronic</td>
    </tr>
    <tr>
      <th>4</th>
      <td>32.0</td>
      <td>0.00465</td>
      <td>0.638</td>
      <td>222369.0</td>
      <td>0.587</td>
      <td>0.909000</td>
      <td>F#</td>
      <td>0.1570</td>
      <td>-6.266</td>
      <td>Major</td>
      <td>0.0413</td>
      <td>145.036</td>
      <td>0.323</td>
      <td>Electronic</td>
    </tr>
    <tr>
      <th>5</th>
      <td>47.0</td>
      <td>0.00523</td>
      <td>0.755</td>
      <td>519468.0</td>
      <td>0.731</td>
      <td>0.854000</td>
      <td>D</td>
      <td>0.2160</td>
      <td>-10.517</td>
      <td>Minor</td>
      <td>0.0412</td>
      <td>?</td>
      <td>0.614</td>
      <td>Electronic</td>
    </tr>
    <tr>
      <th>6</th>
      <td>46.0</td>
      <td>0.02890</td>
      <td>0.572</td>
      <td>214408.0</td>
      <td>0.803</td>
      <td>0.000008</td>
      <td>B</td>
      <td>0.1060</td>
      <td>-4.294</td>
      <td>Major</td>
      <td>0.3510</td>
      <td>149.995</td>
      <td>0.230</td>
      <td>Electronic</td>
    </tr>
    <tr>
      <th>7</th>
      <td>43.0</td>
      <td>0.02970</td>
      <td>0.809</td>
      <td>416132.0</td>
      <td>0.706</td>
      <td>0.903000</td>
      <td>G</td>
      <td>0.0635</td>
      <td>-9.339</td>
      <td>Minor</td>
      <td>0.0484</td>
      <td>120.008</td>
      <td>0.761</td>
      <td>Electronic</td>
    </tr>
    <tr>
      <th>8</th>
      <td>39.0</td>
      <td>0.00299</td>
      <td>0.509</td>
      <td>292800.0</td>
      <td>0.921</td>
      <td>0.000276</td>
      <td>F</td>
      <td>0.1780</td>
      <td>-3.175</td>
      <td>Minor</td>
      <td>0.2680</td>
      <td>149.94799999999998</td>
      <td>0.273</td>
      <td>Electronic</td>
    </tr>
    <tr>
      <th>9</th>
      <td>22.0</td>
      <td>0.00934</td>
      <td>0.578</td>
      <td>204800.0</td>
      <td>0.731</td>
      <td>0.011200</td>
      <td>A</td>
      <td>0.1110</td>
      <td>-7.091</td>
      <td>Minor</td>
      <td>0.1730</td>
      <td>139.933</td>
      <td>0.203</td>
      <td>Electronic</td>
    </tr>
  </tbody>
</table>
</div>



## Section 0.5 Handling Null/String & Categorical Data

Below, we check the presence of `NA`'s and any non-numeric values in the dataframe and afterwards output a description of the dataframe.

First, we create a `lambda` function to attempt to `coerce` all columns into a `numeric` format, and if that is impossible, they are instead turned into `NaN`'s. Furthermore, incorrect (missing) values in the `duration_ms` column are represented as `-1` and also converted into `NaN`'s.

This was done to convert all potentially non-numeric values (entries that cannot be worked with) into a consolidated format.

Next, we impute using the `sklearn.impute.KNNImputer` to find missing values given other available labels in a sample. This incorporates all of the converted `NaN` values in the previous step.

This was done to numerically solve for missing values, using a k-Nearest Neighbor imputation technique, given the dearth of missing data (Section 0.3) that makes this technique particularly potent for the case.


```python
lambda_apply = lambda x: pd.to_numeric(x, errors='coerce')
df[['popularity', 'acousticness', 'danceability', 'duration_ms', 
    'energy', 'instrumentalness', 'liveness', 'loudness',
    'speechiness', 'tempo', 'valence']] = df[['popularity', 'acousticness', 'danceability', 'duration_ms', 
                                              'energy', 'instrumentalness', 'liveness', 'loudness',
                                              'speechiness', 'tempo', 'valence']].apply(lambda_apply).replace(-1, np.NaN)
```


```python
to_impute = df[['popularity', 'acousticness', 'danceability', 'duration_ms', 
                'energy', 'instrumentalness', 'liveness', 'loudness',
                'speechiness', 'tempo', 'valence']]

imputer = KNNImputer(n_neighbors=5)
impute_fit = imputer.fit_transform(to_impute)

df_drop = df.copy()

df_drop[['popularity', 'acousticness', 'danceability', 'duration_ms', 
         'energy', 'instrumentalness', 'liveness', 'loudness',
         'speechiness', 'tempo', 'valence']] = impute_fit

df_drop.dropna(inplace=True)
```

## Section 0.6 `key` & `mode` Column Combination

Below, we create a combined column out of `key` and `mode`, signifying the gradable key, mode (e.g. `A#_Major` or `C#_Minor`) combination in music, denoting the colloquial `key` a song is written in.

This was done in obervance of classical music theory and application of the Circle of Fifths[<sup>[1]</sup>](#fn1) showcasing the cyclical continuity of the `key` feature explored in this analysis.


```python
df_drop['key'] = df_drop['key'] + '_' + df_drop['mode']
df_drop = df_drop.drop(labels=['mode'], axis='columns').reset_index(drop=True)
```

We then check the number of `NaN`' there are in a given column.



```python
df_drop['tempo'].isnull().sum()
```




    0



## Section 0.7 `LabelEncoder` Application to Obtain Numeric Feature Classes

In this section we use the `LabelEncoder` package from `sklearn.preprocessing` to obtain numerical data for features with string values that are continuous. We fit `key` (modified from the previous section) and the label classes in `music_genre` against `LabelEncoder`s.

This was done to enable the `PCA` and `XGBoost` algorithms to be applied later on the be usable.


```python
key_le = LabelEncoder().fit(df_drop['key'])
key_encode = pd.DataFrame(data=key_le.transform(df_drop['key']), columns=['key'])

genre_le = LabelEncoder().fit(df_drop['music_genre'])
genre_encode = pd.DataFrame(data=genre_le.transform(df_drop['music_genre']), columns=['music_genre'])

key_inverse = lambda x: key_le.inverse_transform(x)
genre_inverse = lambda x: genre_le.inverse_transform(x)
```

In the following line we create a copy of `df_encode` and replace the original `key` & `music_genre` columns with the `LabelEncode` transformed outputs.


```python
df_encode = df_drop.copy()

df_encode['key'] = key_encode['key']
df_encode['music_genre'] = genre_encode['music_genre']
```

We then inspect the entire dataframe to determine any missing (`NaN`) values. There fortunately seems to be none.


```python
print(key_encode.isnull().sum())
print(genre_encode.isnull().sum())

df_encode.isnull().sum()
```

    key    0
    dtype: int64
    music_genre    0
    dtype: int64
    




    popularity          0
    acousticness        0
    danceability        0
    duration_ms         0
    energy              0
    instrumentalness    0
    key                 0
    liveness            0
    loudness            0
    speechiness         0
    tempo               0
    valence             0
    music_genre         0
    dtype: int64



## Section 0.8 Train Test Split Using Stratification

In the section below we utilize `train_test_split` from `sklearn.model_selection` to obtain a test size `0.1` the size of the entire dataframe, and stratified against each class of the label (`y = df_encode['music_genre']`). This means we obtain proportional `0.1` amounts from each class label as opposed to obtaining from the total dataset, where it would be highly unlikely to obtain similar proportions of each class.

We do this in accordance to the Spec Sheet, but also to reduce sampling bias given the lack of equality in presence of all classes.


```python
X = df_encode.loc[:,:'valence']
y = df_encode.loc[:,'music_genre']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y)
```

## Section 0.9 Standardization & Normalization

For this dataset, (and the applicable functions of dimensionality reduction through PCA and clustering through k-means), I have chosen to standardize through the below mapping:

$$ \vec{X}_{\textrm{Z-Score}} = \frac{\vec{X} - \mu_{\vec{X}}}{\sigma_{\vec{X}}} $$

where each $\vec{X}$ is a given column of data.


```python
X_train = st.zscore(X_train)
X_test = st.zscore(X_test)
```

# Section 1.0 PCA Fitting and Principal Component Analysis

We run a Principal Component Analysis using the `PCA` package from the `sklearn` library. When, we determine the `eig_vals` as the amount of explained variance, the `loadings` as the components of the output and `rotated_data` as the transformed data. Note the `* -1` for all lines of code: this was done due to a fundamental mistake in Python and makes outputs more interpretable. We then cycle through `eig_vals` to determine the percent of variance explained by each principal component.

This was done to initialize the PCA (with correct assumptions inclusive of Python flaws) and display the amount of variation attributable.


```python
ordinal = lambda n: "%d%s" % (n,"tsnrhtdd"[(n//10%10!=1)*(n%10<4)*n%10::4])

pca, pca_t = PCA().fit(X_train), PCA().fit(X_test)

eig_vals, eig_vals_t = pca.explained_variance_, pca_t.explained_variance_
loadings, loadings_t = pca.components_ * -1, pca_t.components_ * -1
rotated_data, rotated_data_t = pca.fit_transform(X_train) * -1, pca_t.fit_transform(X_test) * -1

covar_explained, covar_explained_t = eig_vals / sum(eig_vals) * 100, eig_vals_t / sum(eig_vals_t) * 100

for n in range(len(covar_explained)):
    explained = covar_explained[n].round(5)
    print(f'{ordinal(n+1)} PC explains {explained}% of variance')
```

    1st PC explains 31.04849% of variance
    2nd PC explains 11.28439% of variance
    3rd PC explains 8.84077% of variance
    4th PC explains 8.32752% of variance
    5th PC explains 8.04052% of variance
    6th PC explains 7.78231% of variance
    7th PC explains 6.787% of variance
    8th PC explains 6.03472% of variance
    9th PC explains 4.94202% of variance
    10th PC explains 3.85229% of variance
    11th PC explains 2.14013% of variance
    12th PC explains 0.91984% of variance
    

## Section 1.1 Scree Plot

Below we show a scree plot of the eigenvalues and the amount of total explainability (out of the total number of `eig_vals`) attributed to each PC. We then draw a line at `Eigenvalue = 1` to show how many `eig_vals` are above 1.

This was done to visually inspect the scree plot and determine the cutoff point for the Kaiser criterion (`Eigenvalue = 1`).


```python
num_col = len(X_train.columns)
x = np.linspace(1,num_col, num_col)
plt.bar(x, eig_vals, color='gray')
plt.plot([0,num_col], [1,1], color='orange')
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue')
plt.title('musicData.csv Principal Components')
plt.show()
```


    
![png](images/Sunny%20Son%20IML%20Capstone_30_0.png)
    


## Section 1.2 Kaiser, Elbow, 90% Variance

Below we show code regarding the number of PCs to choose around each criterion. 

  * For the Kaiser criterion, we count the number of PCs with `eigenvalue > 1`. This was determined to be3 PCs.
  * For the Elbow criterion I eyeballed the data and determined the cutoff point at 2 PCs
  * For the number of factors to account for more than 90% variance of the data, we ran `np.cumsum` on the `covar_explained` (as defined in Section 1.0) and count the number needed. This was determined to be 9 PCs

We do so due to these being the main methods of determining the number of PCs to choose, and run three methods to exhaustively optimize for the number.

Code from Introduction to Data Science, Fall 2021


```python
threshold = 1

print('Number of factors selected by Kaiser criterion:', np.count_nonzero(eig_vals > threshold))

print('Number of factors selected by Elbow criterion: 2')

threshold = 90
eig_sum = np.cumsum(covar_explained)
print('Number of factors to account for at least 90% variance:', np.count_nonzero(eig_sum < threshold) + 1)
```

    Number of factors selected by Kaiser criterion: 3
    Number of factors selected by Elbow criterion: 2
    Number of factors to account for at least 90% variance: 9
    

## Section 1.3 Principal Component Loadings Plots

Below (two cells) we cycle through the first 9 PCs and plot each attributable loadings using `seaborn`. Next, we output each predictor and the associated label.

This was done to provide an initial visual inspection of the loadings plots of each PC.


```python
look_at = 9
for i in range(0,look_at):
    
    plt.subplot(3,3,i+1)
    plt.bar(x, loadings[i,:])
    
    plt.title(f'Principal Component {i+1}', fontsize=10)
    plt.xlabel('Predictor', fontsize=8)
    plt.ylabel('Loading', fontsize=8)
    
    plt.tight_layout()
```


    
![png](images/Sunny%20Son%20IML%20Capstone_34_0.png)
    



```python
for index in range(len(X_train.columns)):
    print(f'Predictor {index+1}: {X_train.columns[index]}')
```

    Predictor 1: popularity
    Predictor 2: acousticness
    Predictor 3: danceability
    Predictor 4: duration_ms
    Predictor 5: energy
    Predictor 6: instrumentalness
    Predictor 7: key
    Predictor 8: liveness
    Predictor 9: loudness
    Predictor 10: speechiness
    Predictor 11: tempo
    Predictor 12: valence
    

## Section 1.4 Scatterplot

We scatter a 2d plot of the first 2 PCs against each other, with PC1 on the x-axis and PC2 on the y-axis. We then color the scattered points associated with the appropriate label class.

This provides an ultimate visual representation of the associated PC loadings for later visual/clustering analysis, as well as a visual (colored) representation of the label calss each point belongs in.

Code modified from Introduction to Data Science Fall 2021


```python
sns.scatterplot(x=rotated_data[:,0], y=rotated_data[:,1], hue=y_train, s=10, marker='x', palette='Paired')

plt.title('Comparison of First Two PCs')
# Set x-axis label
plt.xlabel('Principal Component 1')
# Set y-axis label
plt.ylabel('Principal Component 2')
```




    Text(0, 0.5, 'Principal Component 2')




    
![png](images/Sunny%20Son%20IML%20Capstone_37_1.png)
    


## Section 1.5 3-D Plot of First 3 Principal Components.

We plot in 3d projections using `seaborn`'s `Axes3D` package the first 3 PCs. 

This was done to obtain a better understanding of the data. We see that PC3 does not seem to provide added variance, but rather a distribution horizontally around equal above and below the mean. However, in the 3rd dimension striations seem to result from the label classes presented.


```python
%matplotlib widget

fig = plt.figure()
ax = fig.gca(projection='3d')
fig.add_axes(ax)

cmap = ListedColormap(sns.color_palette('Paired', 10).as_hex())

sc = ax.scatter(rotated_data[:,0],
                rotated_data[:,1],
                rotated_data[:,2],
                s=40, marker='.', c=y_train.to_list(), cmap=cmap, alpha=0.05)

ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')

ax.legend(*sc.legend_elements(), bbox_to_anchor=(1.05, 1), loc=2)

plt.show()
```

    /var/folders/75/nxb38mxd16q4m2ylbprp28s00000gn/T/ipykernel_78524/1430082577.py:4: MatplotlibDeprecationWarning: Calling gca() with keyword arguments was deprecated in Matplotlib 3.4. Starting two minor releases later, gca() will take no keyword arguments. The gca() function should only be used to get the current axes, or if no axes exist, create new axes with default keyword arguments. To create a new axes with non-default arguments, use plt.axes() or plt.subplot().
      ax = fig.gca(projection='3d')
    



<div style="display: inline-block;">
    <div class="jupyter-widgets widget-label" style="text-align: center;">
        Figure
    </div>
    <img src='data:image/png;base64,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n59PRkYGHR0dvPDCC89IKU8Y41O2icEWsImBfRHj0NLSwvz58xFCsGTJEnRdNy2W8vLyki5mKufW09NDeXm56Y5hncSc7H63ZImXx+OhsbGRqqqqKIeL8UI4HDYj8T/84Q988MEHnH322dx1113Mnj2bQw89FI/HY7rDnHLKKRrsmAM2pidvY2IL2MTAvohx6OzsZOXKlSxatIj09HR8Pp9pNpzsmWaqmdfv9/cxJVYl66owQlVV5ubmJnR5MVni1draarYADNYMnGxWrFjBrbfeatp/HXrooVx88cVD3n/58uX09vayZMkSJk+ejN/v57zzzqOsrIzrrrvOuqlmi9fOhy1gEwP7Ig4TNdOstbWV7u7uhM40UwUjoVCIqVOnDphzizXhtU6cHk2hR7LES/lITp06Nam2X0Pl+uuv5+tf/zoLFiwY0vbWyAvgvPPO4/XXX2fFihXsscceAHz11Vfcd9993HzzzdYvFKl1drYZEraATQzsizgKEjnTLBwOU1dXBzDsWWKqslL1mo3EUR6SJ17t7e3U19dTXl5OWVlZwo47GhYtWkRubi5NTU0IIbj66qv7baC2itfnn3/OXnvthaZpXHPNNfzrX//iscceo6ysjBtuuAGPx8Ntt0VVytsCthNiC9jEwL6ICWI0M83U3Ku0tDQqKipGHcWpnJlqnM7LyxuwmRiSJ17KdqusrIzy8vKEHXe0nHfeeZx++ul84xvf4LbbbqO+vp5bb711wH1++9vf8uKLL1JdXc1PfvITDjvsMK699lqeeuopjjjiCAoLC7n88stxuVzWQg9bwHZC7DL6XYyWlhZOP/1087HX66W1tZVPPvkkaru6ujoWLFigemAoKiriT3/6U0rPdSxIT0+npKSEkpKSqJlm9fX1Zul+vHliapaZy+VKyCDMWEd51UxcX19POByO2zidLPFSno2lpaVjJl4vv/yy2VismDFjBg8++KD5+Mwzz+Twww/vs6818lq1ahU1NTU899xz3HvvvaxcuZJgMMi1117L5MmTuf/++83etkAgMCYtDzZDx47AJgYjuojhcJhTTz2VRYsWcfTRR0c998orr/DOO+/EJrJ3WdRMs9bWVjMiUmLW0tLCp59+yoEHHkhpaWnSy/RVT1lHR4fpjOF0OvF4PFRWViZUvDo7O6mtraW4uJjKysqEHTcReL1ennnmGU477TTAyM/NmzeP9957L+72K1euZOvWraSlpXHuuecSDAa5++67Wb9+Pd/5znf4wQ9+wA033MCjjz7K66+/TkVFhXV3OwLbCbHd6HdhnnnmGTIzM/uIFxg5gq+++opjjjmGU045BSnlGJzhzoOaaTZr1iz22Wcfpk2bhsPh4P333+enP/0p7777Lm6323QyTyaqaXrGjBlMmzYNXddpbm4GjArBjo4OwuHRT/RQ4lVUVLRTNidnZWXxxz/+kc8++wwwzHetEZj1y/lf//pX7rnnHjZt2sSTTz7Jq6++itPp5KKLLmLq1KmsWbMGgKuuuopf//rXseJls5NiR2ATg2FfxFAoxBFHHME999yDEKLP83fddReFhYWceOKJ/Otf/+L666/npZdeGhMLpJ2Vzs5Ovvvd77Lffvtx4YUX4vP5CIfDZGdnp6THC4woZNu2bVRWVuJ0Os2cmc/nM89jJI3TXV1dbN26lcLCQqqqqlLW/D1cPvzwQ2688UZ8Ph/Tpk3jlltu6TPC5emnn2bz5s2cfvrp5OTk8MQTT/DKK6/w4x//mPnz5wNGVDuIN+XO+QHs4tgCNjGIexEHyhu89dZbPPLII0POay1cuJBbbrmF3XffffRnO0EIh8O88cYbHHrooTidTsLhsOm80dHRQSgUIisrKyk9XhAtXrEjWYLBoNlrphqnVVXlYKLa3d3N1q1bmTx5MtXV1TuteA2GsolatGgRX3zxBStXrmTmzJk0Njby2muv8fjjj3PhhRdyxBFHAINOaR6fH8IExxawicGwL+JVV13F17/+dX7wgx/Eff6RRx5hwYIFpjnr0Ucfze23385uu+02ujPdRUj2TLOBxCuWUChkillXV9eAjdM9PT3U1NSQn5/P9OnTx514WUWos7PTHBR6/vnnU1NTw5NPPklmZiaNjY289957HHrooUMd/TK+PohdBLsKcRfl008/jTvQT/HBBx/g8/k466yzWLVqFeFwmBkzZqTwDMc3mqaRn59Pfn5+1Ewzj8dDU1PTqGaaDUe8wJh8rc7F2ji9fft208G/u7ubSZMmUV9fT15e3rgUL9gxRXnlypX885//JBwOs+eee3L33XdzwQUXsGjRIh5//HGmTJnCMcccg6ZpfZqbbcYPdgQ2MYi6iNZvnv2xzz77sGrVqqh1/8cff5ympiaWLFlCY2MjS5cuNW9yN954o718mAB0Xaezs9OMzIY702y44jXYuXR1ddHW1saPf/xjdF3nW9/6Fj/60Y84+OCDR3XsseSll17ivvvu46qrrmL9+vV8+OGHZGRkcNNNN3HiiScCxt/6MAV6/Kn5LoAtYBMD8yJ2dnbyve99j5kzZ3LkkUdyxBFHUFpamrITGYo3nd/v58orr2TNmjVkZGTw29/+lpkzZ6bsHHcWYmea+f3+AWeaJVK8rPh8Pv773//y+eefs3r1aj7++GOef/75ncIqarh0d3ezdOlSjj32WL773e8SCASQUnLjjTdy0003MX36dBobG0fy3mwB2wmxlxAnGB9++CFut5tf/vKXrFixgvPPP5/s7GyuueaalCwBrlmzhqVLlw7oTffII4+QmZnJyy+/zAcffMAvf/lL/vrXvyb93HY2YmeaWcfAxM40e+uttygqKmKvvfZKqHj19vZSU1PDlClTOOigg1I65iURbN++ncbGRmpqajj00EPJysoiJyeHnp4ewGhM32uvvcjKyqKhoYHp06dTUlJiLxtOEOwrOMF4/vnnOfjgg9l7771ZtmwZTz/9NAcccACPP/44YHzrT0SPUH98/vnnrFixgqOPPppLL72U9vb2Ptu89dZbLFy4EID9998fj8dDfX190s5pPKDEbOrUqey1117sscceFBQU0NXVxZ133slVV11FV1cXEN3fNBqUeGVmZjJz5swxE6877riDu+66y3zc0dHBz372M+bNm8dJJ53E9u3b4+63adMmzjnnHO6++25eeOEFPv74YwAKCwu57777qKurIxgMsmrVKhobG81GbE3TbPGaINhXcQLR09PDqlWrzHV+hd/vp6GhgVAoFPWPt7Ozk//85z80NjYm7ByKi4s599xzee655+KNpACgqakpaoZUcXExDQ0NCTuHiUBWVhYVFRVs2rSJp59+mquuuorq6mo2b97Mhg0baGhooLu7e8Ri5vf7qampwe12s9tuu42JeHm9Xq644gr+7//+L+r3d9xxB3PnzuXll1/mhz/8ITfeeGOffevr67nssss4+eSTue+++7jtttvYf//9Abjkkks44IADOOecc7j44ou55ZZbuP7665k6dWrCxN9m58BeQpxAfPrpp2zfvp0bb7yRffbZh+9///sUFBTw7LPPcuGFF/LPf/6TDRs2UFFRwbe//W1yc3Pp7e1l7dq1TJky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NlFUVVUxffp0AL73ve8xZcoUfvWrX/HrX/8aIQQ+n4+tW7dy2GGH8fHHH5OVlcUBBxwAwOTJk/H7/dTW1vL666/zzjvvUF5ezp133snixYtNZ4TPPvuMzz77rM/odxjc5++yyy6jurqaO+64o985Wh988AFPP/00gD3LbBzicDhobW1lyZIlzJo1izVr1nDOOefg9/tNEfvZz35Ge3v7oH8vNhMb+2upTb/k5OSwZMkSwDBYzcvL4y9/+QvTpk3D5XLxySefUF9fT0FBAQDr1q2jqamJ3Xffnblz5zJt2jSmTJnCySefzPHHH8/nn39ORUUF//nPf8wx7rHLQJqmsXbtWoqLi828nOKLL77g9ddfZ7fddjNdFkpKSnjggQei+syuvPJKli5dysqVK3G73dx66622C8M4IhwOc/PNN3P00Udz0kknsWHDBrO366677uK0005j/vz55Ofnj/Wp2owxdg5sYpCUi6jr/Tcur169mj/96U80NjYyd+5cJk2axH/+8x/2339/5s6dywUXXMCCBQvIzs6mpqaGF198kRdffBFN07jooovMnFQ4HDbL79966y1effVVenp62Lx5M/n5+cybN4/jjjuO9PT0pCbmn3/+ee69916CwSCnnnoqJ510UtTzX375JVdeeSVdXV3MnTuXX/3qV/ayZAJRuS4w2jiWLl3KKaecwrp169A0jWOPPZYjjzySSZMmceutt5qrBNb9koydA9sJsb+W2vRLbEM07Fjie+edd3C5XCxbtoyioiI2b97MGWecwdlnn81LL73EUUcdxRVXXMGSJUuYO3cus2bNorKyko8++oisrCyzoCItLQ1N02hsbOSKK65g9uzZ3HDDDaxYsYILLriAN954g/feew9N05K2XNTY2Mjtt9/OX/7yF/72t7/x5JNPsn79+qhtLrvsMpYtW8Yrr7yCruu75PyyoRA7GmXVqlUccMABHHPMMRxzzDH88pe/7LOP6iMMh8Pouk5GRgZnn30206dP54svvuCAAw6gp6eHOXPmsHjxYlO8wK423NWxv0LaDAtN0wgGg0ybNg2Hw8Gee+7JnnvuGbXN3nvvzf3338+yZcv42te+xu9//3tOOOEEnE4nq1at4pvf/CaA6X7f3d3NypUrmTVrFqeddppZdr/ffvtx4IEH8thjj7HvvvuaS0bqRpeoJP67777Lt771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' width=432.0/>
</div>




```python
%matplotlib inline
```

## Section 1.6 Results & Discussion

From the Section 1.1 scree plot & 1.3 analytical determination, we see that 4 PCs have `eigenvalue > 1`(adhering to the Kaiser criterion). This means that the number of PCs that have greater explanability of the total dataset greater than an individual feature is 3.

From the Section 1.0 original fitting and `cumsum` of `eig_vals` analysis, we see that the first PC accounts for "`31.04386%` of variance" and the second PC accounts for "`11.29876%` of variance" (the third PC accounting for `8.84217%`). Together, they (the first two) account for `42.34262%` of the variance in the data (with 3 PCs accounting for `51.18479%`). This means that with just 2 PCs, almost **half** of the variance in the data is accounted for (and with 3 PCs, slightly more than half is accounted for). In 2-D this would be a well-suited method for visual/future-clustering analysis, but in scope of this analysis (applying the Kaiser)

From viewing the `loadings` of the first two PCs, we see that:

   * PC1: loadings points very positively towards `popularity`, `danceability`, `energy`, `loudness`, and `valence` but very negatively towards `acousticness` and `instrumentalness`. Unlisted loadings were smaller/near-zero in magnitude.
   
   
   * PC2: strongly positive towards `popularity`, `acousticness`, `danceability`, `speechiness`, negatively towards `duration_ms`, `energy`, `liveliness`, `loudness`, and `tempo`. Unlisted loadings were smaller/near-zero in magnitude.
   
   
In interpretation, we see that PC1 is accute towards features able to be interpreted as having qualities of a "pop song" (with `popularity`, `danceability`, and `loudness` all influencing the quality), and PC2 as being accute in features positioned toward dance music: `danceability`, `popularity`, and `speechiness`.

# Section 2.0 K-Means Clustering

For this Section, we realized the importance of using a distance-preserving method of dimensionality reduction, given k-means' dependance on distance metrics to create clusters. This is why I chose PCA as the form of dimensionality reduction to be used, as opposed to t-SNE, which only preserves *local* similarities. I did not choose MDS nor t-SNE due to the lack of open interpretability of factors in both (loadings).

In the two blocks of code below, we call and create a `np.array` of the first two PCs of the PCA analysis done in Section 1. We then create a `np.array` of `NaN`s of `num_clusters` to look at, with `num_clusters=9` to store each sum of silhouette scores for every number of clusters tried.

We then loop through the 9 total clusters, using the `KMeans` class from `sklearn.cluster` to calculate the clusters for each number of clusters, and create subplots using `matplotlib` of associated silhouette score (found using `silhouette_samples` from `sklearn.metrics`) vs. count of occurrence. Each plot displays the number of clusters and the sum of the silhouette scores as the title.

This was done to output subplots showing the silhouette score vs. the count of occurrence, with titles bearing the sum of silhouette scores.

The code below is modified from Introduction to Data Science, Fall 2021


```python
x = np.column_stack((rotated_data[:,0], rotated_data[:,1]))

num_clusters = 9 # how many clusters are we looping over? (from 2 to 10)
Q = np.empty([num_clusters,1])*np.NaN # init container to store sums
```


```python
# Compute kMeans:
for i in range(2, 11): # Loop through each cluster (from 2 to 10!)
    kMeans = KMeans(n_clusters = int(i)).fit(x) # compute kmeans using scikit
    cId = kMeans.labels_ # vector of cluster IDs that the row belongs to
    cCoords = kMeans.cluster_centers_ # coordinate location for center of each cluster
    s = silhouette_samples(x, cId) # compute the mean silhouette coefficient of all samples
    Q[i-2] = sum(s) # take the sum
    # Plot data:
    plt.subplot(3, 3, i-1)
    plt.hist(s,bins=20) 
    plt.xlim(-0. 2,1)
    plt.ylim(0, 10000)
    plt.xlabel('Silhouette score')
    plt.ylabel('Count')
    plt.title(f'Clust: {i}, Sum: {int(Q[i-2])}') # sum rounded to nearest integer
    plt.tight_layout() # adjusts subplot padding
```


    
![png](images/Sunny%20Son%20IML%20Capstone_44_0.png)
    


## Section 2.1 Plot of Number of Clusters vs. Sum of Silhouette

Below we show on the x-axis the number of clusters vs. the sum of silhouette scores on the y-axis. We use `matplotlib` to plot the associated `np.linspace` object, set between 2 to 10 with 9 in-between divisions against the sum of silhouette scores obtained from the filled `Q` array in the previous section.

The code below is modified from Introduction to Data Science, Fall 2021


```python
plt.plot(np.linspace(2,10,9), Q)

plt.xlabel('Number of Clusters')
plt.ylabel('Sum of Scores')
plt.title('Sum of Silhouette Scores per Cluster Number')

plt.show()
```


    
![png](images/Sunny%20Son%20IML%20Capstone_46_0.png)
    


## Section 2.2 Plot of First Two PCs

Below we plot the the first two PCs, colored by individual clusters (the number of clusters obtained by optimal silhouette value obtained above through `np.argmax` of the `Q` silhouette sum array added to 2 (`np.argmax(Q) + 2` because we start at index 0 to store the lowest number of clusters, 2). We first hard code the optimal number of clusters into `num_clusters`, obtained from the silhouette plot (Section 2.1) above (which is `2`). We instance a `KMeans` class object and fit it with *two* dimensions of the `loadings` from PCA analysis (Section 1), as requested by the spec sheet to produce visualizations in 2-D. Next, looping through the number of clusters to be plotted (`num_clusters`), we plot (in `green`, `blue`, `orange` respectively), associated clusters by determining the indices of associated labels in `rotated_data`. Finally, we plot the associated centroids.

This was done to visualize in two PC dimensions individual clusters, alongside their associated centroids.

*Note*: if code does not output clusters ALONGSIDE centroids, please disregard this section in grading and use the 3-D analogy of Section 2.3 instead.

Code adapted from Introduction to Data Science, Fall 2021


```python
%matplotlib inline
num_clusters = np.argmax(Q) + 2
kMeans = KMeans(n_clusters=num_clusters)
kMeans = kMeans.fit(np.column_stack((rotated_data[:,0], rotated_data[:,1])))

cId = kMeans.labels_
cCoords = kMeans.cluster_centers_

for i in range(num_clusters):
    plotIndex = np.argwhere(cId == int(i))
    plt.plot(rotated_data[plotIndex,0], rotated_data[plotIndex,1], '.', markersize=10, alpha=0.2)
    plt.plot(cCoords[i,0], cCoords[i,1],'o', markersize=10, color='black', alpha=1.0)
    
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Comparison of First Two PCs')
```


    
![png](images/Sunny%20Son%20IML%20Capstone_48_0.png)
    


## Section 2.3 3-D Plot of 3 Principal Component Clustering

This section maps the two chosen optimal number of clusters onto a 3-D space with 3 PCs. 

This was done to visualize what k-means would look like in higher-dimensions with more variation in the data (explained by more PCs) included.


```python
%matplotlib notebook

num_clusters = np.argmax(Q) + 2
kMeans = KMeans(n_clusters = num_clusters).fit(np.column_stack((rotated_data[:,0],
                                                                rotated_data[:,1],
                                                                rotated_data[:,2]))) 
cId = kMeans.labels_ 
cCoords = kMeans.cluster_centers_ 

fig = plt.figure(figsize=(6,6))
ax = Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)

# Plot the color-coded data:
for i in range(num_clusters):
    
    plotIndex = np.argwhere(cId == int(i))
    
    sc = ax.scatter(rotated_data[plotIndex,0],
                    rotated_data[plotIndex,1],
                    rotated_data[plotIndex,2],
                    label=f'Cluster {i+1}',
                    s=40, marker='.', alpha=0.05)
    
    sc = ax.scatter(cCoords[int(i-1),0],
                    cCoords[int(i-1),1],
                    cCoords[int(i-1),2],
                    color='black', s=70, alpha=1.0)
    
ax.legend(loc='best')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
```

    Warning: Cannot change to a different GUI toolkit: notebook. Using widget instead.
    




    Text(0.5, 0, 'Principal Component 3')





<div style="display: inline-block;">
    <div class="jupyter-widgets widget-label" style="text-align: center;">
        Figure
    </div>
    <img src='data:image/png;base64,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' width=432.0/>
</div>




```python
%matplotlib inline
```

## Section 2.4 Results & Discussion

From the above, we find that by implementing k-means on a PCA reduced `musicData.csv`, with a sum of silhouette scores of `24558`, and an optimal number of clusters of `2` (Sections 2.0 & 2.1), we obtain the visualization shown in Section 2.2.

This however, could *potentially* not be accurate given a brief viewing of the dataset: shown in Section 2.1, we can see isses with the dataset being potentially unable to be broken up into individual "spherical clusters" (due to the singular, already clustered blob resulting from the PCA). Potential outliers are also included in the k-means analysis, such as the point around (`-3.9`, `-15.25`) in the graph of Section 2.2.

However, this could also mean that using the first *two* and *three* PCs respectively, explaining `42.34262%` and `51.18479%` of the variation in the data respectively (Section 1.6), `2` clusters *do* exist, allowing for further analysis through unsupervised classification (not proceeded with in the scope of this analysis).

# Section 3.0 Xtreme Gradient Boost (`XGBoost`) Function Definitions

Below we create a dictionary of preprocessed `training_data` inclusive of `X_train`, `X_test`, `y_train`, & `y_test`.

This was done for ease of accessibility.

In the next block of code, we define `fitXgb` to output the fitting progress (`mlogloss` and `merror`, `m` meaning multiclass for each `test` and `train` sets) every 100 epochs. At the end of fitting, we output the `accuracy`, `micro f1` score, and `macro f1` score.[<sup>[4]</sup>](#fn4)

Each `micro` and `macro` score utilize different aggregation methods of determining a combined score, with `micro` calculating each precision, recall by aggregating then dividing, and `macro` calculating each precision and recall individually, then finding the combined average.

We then define `plot_compare` to output visualization plots of the `mlogloss` and `merror` for each `train` and `test` set.

This was done to give more quantitative (for the former) and qualitative (latter plot) information regarding the fitting of the `XGBoost` model.

I choose `XGBoost` due to it being the "silver bullet" of Machine Learning at the moment, as well as the combinatorially positive effects of mixing together the plus-sides of random forests (bagging) as well as the adherence to an objective (loss) function through "Gradient Boosting." Gradient Boosted Decision Trees (GBDTs) formalizes additively generating weak models (akin to AdaBoost) but over an objective function. Each successive, iterative model trains a "shallow decision tree,.. over the error residuals of a previous model to fit the next." The model then weighs the sum of all predictions.[<sup>[5]</sup>](#fn5)


```python
training_data = {'X_train': rotated_data[:,:4],
                 'X_test': rotated_data_t[:,:4],
                 'y_train': y_train,
                 'y_test': y_test}
```


```python
#allow logloss and classification error plots for each iteration of xgb model
def plot_compare(metrics, eval_results, epochs):
    for m in metrics:
        test_score = eval_results['test'][m]
        train_score = eval_results['train'][m]
        rang = range(0, epochs)
        plt.rcParams["figure.figsize"] = [6,6]
        plt.plot(rang, test_score,"c", label="Test")
        plt.plot(rang, train_score,"orange", label="Train")
        title_name = m + " plot"
        plt.title(title_name)
        plt.xlabel('Iterations')
        plt.ylabel(m)
        lgd = plt.legend()
        plt.show()
        
def fitXgb(sk_model, training_data=training_data, epochs=100):
    print('Fitting model...')
    sk_model.fit(training_data['X_train'], training_data['y_train'].values.reshape(-1, 1))
    print('Fitting done!')
    train = xgb.DMatrix(training_data['X_train'], label=training_data['y_train'])
    test = xgb.DMatrix(training_data['X_test'], label=training_data['y_test'])
    params = sk_model.get_xgb_params()
    metrics = ['mlogloss','merror']
    params['eval_metric'] = metrics
    store = {}
    evallist = [(test, 'test'), (train,'train')]
    xgb_model = xgb.train(params, train, epochs, evallist, evals_result=store, verbose_eval=100)
    print('-- Model Report --')
    
    accuracy = accuracy_score(sk_model.predict(training_data['X_test']), training_data['y_test'])
    f1_micro = f1_score(sk_model.predict(training_data['X_test']), training_data['y_test'], average='micro')
    f1_macro = f1_score(sk_model.predict(training_data['X_test']), training_data['y_test'], average='macro')
    
    print(f'XGBoost Accuracy: {accuracy}')
    print(f'XGBoost F1-Score (Micro): {f1_micro}')
    print(f'XGBoost F1-Score (Macro): {f1_macro}')

    plot_compare(metrics, store, epochs)
```

## Section 3.1 `XGBoost` Initial Fitting & Analysis

Below, we fit out initial `XGBoost` model, `xgb_clf1` with `objective=softproba` to determine AUC ROC in later sections.

This was done to be able to determine AUC ROC (also, as specified by the spec sheet). We specify which hyperparameters will be tuned later on within the code, and output graphics and text using the functions defined in Section 3.0 above.


```python
## Computer/dataset specifics about running the XGBoost model
num_threads = 8
num_classes = 10
```


```python
#initial model
xgb_clf1 = XGBClassifier(learning_rate=0.1,
                         n_estimators=1000,
                         max_depth=5, #tunable
                         min_child_weight=1, #tunable
                         gamma=0,
                         subsample=0.8,
                         colsample_bytree=0.8,
                         objective='multi:softproba',
                         nthread=num_threads,
                         num_class=num_classes,
                         seed=1234567)
```


```python
fitXgb(xgb_clf1, training_data)
```

    Fitting model...
    Fitting done!
    [0]	test-mlogloss:2.24280	test-merror:0.71080	train-mlogloss:2.23709	train-merror:0.68424
    

    /Users/sunnyson/opt/anaconda3/lib/python3.9/site-packages/xgboost/core.py:617: FutureWarning: Pass `evals` as keyword args.
      warnings.warn(msg, FutureWarning)
    

    [99]	test-mlogloss:1.72184	test-merror:0.67020	train-mlogloss:1.52533	train-merror:0.58136
    -- Model Report --
    XGBoost Accuracy: 0.3096
    XGBoost F1-Score (Micro): 0.3096
    XGBoost F1-Score (Macro): 0.29959849702613905
    


    
![png](images/Sunny%20Son%20IML%20Capstone_59_3.png)
    



    
![png](images/Sunny%20Son%20IML%20Capstone_59_4.png)
    


## Section 3.2 Tuning `max_depth` & `min_child_weight`

Direcly below, we define `getTrainScores` to output more detail about the fully tuned model shown further below.

Two blocks below, we create a `param_test` dictionary of hyperparameters to tune, the two being `max_depth` & `min_child_weight`. We then use `GridSearchCV` from `sklearn.model_selection` to fit different cardinal products of the `param_test` hyperparameters to maximize the `ovr_auc_roc` function. 

The aforementioned loss function is a heuristic for using individual, binary classifications to proxy a multi-class classification on the AUC ROC of the target class versus the combination of all other classes for the label.

This was done to induce hyperparameter tuning of the two mentioned above, but in a way directed towards maximizing AUC ROC. 


```python
def getTrainScores(gs):
    results = {}
    runs = 0
    for x,y in zip(list(gs.cv_results_['mean_test_score']), gs.cv_results_['params']):
        results[runs] = 'mean:' + str(x) + 'params' + str(y)
        runs += 1
    best = {'best_mean': gs.best_score_, "best_param":gs.best_params_}
    return results, best
```


```python
param_test1 = {
 'max_depth': [3, 6, 9],
 'min_child_weight': [1, 3, 5]
}
# metrics to consider: f1_micro, f1_macro, roc_auc_ovr
gsearch1 = GridSearchCV(estimator=xgb_clf1, param_grid=param_test1, scoring='roc_auc_ovr', verbose=10, cv=2)
gsearch1.fit(training_data['X_train'], training_data['y_train'])
```

    Fitting 2 folds for each of 9 candidates, totalling 18 fits
    [CV 1/2; 1/9] START max_depth=3, min_child_weight=1.............................
    [CV 1/2; 1/9] END max_depth=3, min_child_weight=1;, score=0.815 total time= 1.0min
    [CV 2/2; 1/9] START max_depth=3, min_child_weight=1.............................
    [CV 2/2; 1/9] END max_depth=3, min_child_weight=1;, score=0.816 total time= 1.2min
    [CV 1/2; 2/9] START max_depth=3, min_child_weight=3.............................
    [CV 1/2; 2/9] END max_depth=3, min_child_weight=3;, score=0.815 total time= 1.1min
    [CV 2/2; 2/9] START max_depth=3, min_child_weight=3.............................
    [CV 2/2; 2/9] END max_depth=3, min_child_weight=3;, score=0.816 total time=  57.7s
    [CV 1/2; 3/9] START max_depth=3, min_child_weight=5.............................
    [CV 1/2; 3/9] END max_depth=3, min_child_weight=5;, score=0.815 total time=  51.7s
    [CV 2/2; 3/9] START max_depth=3, min_child_weight=5.............................
    [CV 2/2; 3/9] END max_depth=3, min_child_weight=5;, score=0.816 total time= 1.0min
    [CV 1/2; 4/9] START max_depth=6, min_child_weight=1.............................
    [CV 1/2; 4/9] END max_depth=6, min_child_weight=1;, score=0.801 total time= 2.0min
    [CV 2/2; 4/9] START max_depth=6, min_child_weight=1.............................
    [CV 2/2; 4/9] END max_depth=6, min_child_weight=1;, score=0.801 total time= 1.9min
    [CV 1/2; 5/9] START max_depth=6, min_child_weight=3.............................
    [CV 1/2; 5/9] END max_depth=6, min_child_weight=3;, score=0.798 total time= 1.4min
    [CV 2/2; 5/9] START max_depth=6, min_child_weight=3.............................
    [CV 2/2; 5/9] END max_depth=6, min_child_weight=3;, score=0.800 total time= 1.3min
    [CV 1/2; 6/9] START max_depth=6, min_child_weight=5.............................
    [CV 1/2; 6/9] END max_depth=6, min_child_weight=5;, score=0.798 total time= 2.0min
    [CV 2/2; 6/9] START max_depth=6, min_child_weight=5.............................
    [CV 2/2; 6/9] END max_depth=6, min_child_weight=5;, score=0.799 total time= 1.4min
    [CV 1/2; 7/9] START max_depth=9, min_child_weight=1.............................
    [CV 1/2; 7/9] END max_depth=9, min_child_weight=1;, score=0.790 total time= 2.5min
    [CV 2/2; 7/9] START max_depth=9, min_child_weight=1.............................
    [CV 2/2; 7/9] END max_depth=9, min_child_weight=1;, score=0.791 total time= 2.2min
    [CV 1/2; 8/9] START max_depth=9, min_child_weight=3.............................
    [CV 1/2; 8/9] END max_depth=9, min_child_weight=3;, score=0.788 total time= 2.2min
    [CV 2/2; 8/9] START max_depth=9, min_child_weight=3.............................
    [CV 2/2; 8/9] END max_depth=9, min_child_weight=3;, score=0.789 total time= 1.7min
    [CV 1/2; 9/9] START max_depth=9, min_child_weight=5.............................
    [CV 1/2; 9/9] END max_depth=9, min_child_weight=5;, score=0.787 total time= 1.9min
    [CV 2/2; 9/9] START max_depth=9, min_child_weight=5.............................
    [CV 2/2; 9/9] END max_depth=9, min_child_weight=5;, score=0.788 total time= 2.2min
    




    GridSearchCV(cv=2,
                 estimator=XGBClassifier(base_score=None, booster=None,
                                         callbacks=None, colsample_bylevel=None,
                                         colsample_bynode=None,
                                         colsample_bytree=0.8,
                                         early_stopping_rounds=None,
                                         enable_categorical=False, eval_metric=None,
                                         feature_types=None, gamma=0, gpu_id=None,
                                         grow_policy=None, importance_type=None,
                                         interaction_constraints=None,
                                         learning_rate=0.1, max_bin=None,
                                         max_cat_threshold=None,
                                         max_cat_to_onehot=None,
                                         max_delta_step=None, max_depth=5,
                                         max_leaves=None, min_child_weight=1,
                                         missing=nan, monotone_constraints=None,
                                         n_estimators=1000, n_jobs=None, nthread=8,
                                         num_class=10, num_parallel_tree=None, ...),
                 param_grid={'max_depth': [3, 6, 9], 'min_child_weight': [1, 3, 5]},
                 scoring='roc_auc_ovr', verbose=10)




```python
getTrainScores(gsearch1)
```




    ({0: "mean:0.8156157459533608params{'max_depth': 3, 'min_child_weight': 1}",
      1: "mean:0.8152964554183812params{'max_depth': 3, 'min_child_weight': 3}",
      2: "mean:0.8153314858710563params{'max_depth': 3, 'min_child_weight': 5}",
      3: "mean:0.8009061569272977params{'max_depth': 6, 'min_child_weight': 1}",
      4: "mean:0.799073213717421params{'max_depth': 6, 'min_child_weight': 3}",
      5: "mean:0.7985815017832647params{'max_depth': 6, 'min_child_weight': 5}",
      6: "mean:0.7903423593964335params{'max_depth': 9, 'min_child_weight': 1}",
      7: "mean:0.7885236373113854params{'max_depth': 9, 'min_child_weight': 3}",
      8: "mean:0.7873373580246914params{'max_depth': 9, 'min_child_weight': 5}"},
     {'best_mean': 0.8156157459533608,
      'best_param': {'max_depth': 3, 'min_child_weight': 1}})



### Section 3.2.1 Optimal Parameter Selection

Below we select the optimal parameters for `opt_min_child` and ` opt_max_depth` based on hyperparameter tuning done previously.


```python
opt_min_child = 1
opt_max_depth = 3
```

### Section 3.2.2 Output of Model with Optimal `max_depth` & `min_child_weight`

Below we output the model with the determined optimal `max_depth` and `min_child_height` found above. We evoke the functions defined in Section 3.0 to visually and numerically break down `mlogloss` and `merror`.


```python
xgb_clf2 = XGBClassifier(learning_rate=0.1,
                         n_estimators=1000,
                         max_depth=opt_max_depth, # tuned
                         min_child_weight=opt_min_child, # tuned
                         gamma=0,
                         subsample=0.8,
                         colsample_bytree=0.8,
                         objective='multi:softproba',
                         nthread=num_threads,
                         num_class=num_classes,
                         seed=1234567)

fitXgb(xgb_clf2, training_data)
```

    Fitting model...
    Fitting done!
    [0]	test-mlogloss:2.24656	test-merror:0.71280	train-mlogloss:2.24377	train-merror:0.69944
    

    /Users/sunnyson/opt/anaconda3/lib/python3.9/site-packages/xgboost/core.py:617: FutureWarning: Pass `evals` as keyword args.
      warnings.warn(msg, FutureWarning)
    

    [99]	test-mlogloss:1.71226	test-merror:0.67100	train-mlogloss:1.61737	train-merror:0.62084
    -- Model Report --
    XGBoost Accuracy: 0.3162
    XGBoost F1-Score (Micro): 0.3162
    XGBoost F1-Score (Macro): 0.3060725450187838
    


    
![png](images/Sunny%20Son%20IML%20Capstone_67_3.png)
    



    
![png](images/Sunny%20Son%20IML%20Capstone_67_4.png)
    


## Section 3.3 Tuning `reg_alpha`

Direcly below, we define `getTrainScores` to output more detail about the fully tuned model shown further below.

Two blocks below, we create a `param_test` dictionary of hyperparameters to tune, the two being `reg_alpha`. We then use `GridSearchCV` from `sklearn.model_selection` to fit different cardinal products of the `param_test` hyperparameters to maximize the `ovr_auc_roc` function.

The aforementioned loss function is a heuristic for using individual, binary classifications to proxy a multi-class classification on the AUC ROC of the target class versus the combination of all other classes for the label.

This was done to induce hyperparameter tuning of the two mentioned above, but in a way directed towards maximizing AUC ROC. 


```python
param_test2 = {
 'reg_alpha':[1e-2, 0.1, 1]
}

gsearch2 = GridSearchCV(estimator = xgb_clf2, param_grid = param_test2, scoring='roc_auc_ovr', verbose = 10, cv=2)
gsearch2.fit(training_data['X_train'], training_data['y_train'])
```

    Fitting 2 folds for each of 3 candidates, totalling 6 fits
    [CV 1/2; 1/3] START reg_alpha=0.01..............................................
    [CV 1/2; 1/3] END ...............reg_alpha=0.01;, score=0.815 total time=  35.1s
    [CV 2/2; 1/3] START reg_alpha=0.01..............................................
    [CV 2/2; 1/3] END ...............reg_alpha=0.01;, score=0.816 total time=  42.9s
    [CV 1/2; 2/3] START reg_alpha=0.1...............................................
    [CV 1/2; 2/3] END ................reg_alpha=0.1;, score=0.815 total time=  37.3s
    [CV 2/2; 2/3] START reg_alpha=0.1...............................................
    [CV 2/2; 2/3] END ................reg_alpha=0.1;, score=0.816 total time=  34.3s
    [CV 1/2; 3/3] START reg_alpha=1.................................................
    [CV 1/2; 3/3] END ..................reg_alpha=1;, score=0.815 total time=  45.6s
    [CV 2/2; 3/3] START reg_alpha=1.................................................
    [CV 2/2; 3/3] END ..................reg_alpha=1;, score=0.816 total time=  55.2s
    




    GridSearchCV(cv=2,
                 estimator=XGBClassifier(base_score=None, booster=None,
                                         callbacks=None, colsample_bylevel=None,
                                         colsample_bynode=None,
                                         colsample_bytree=0.8,
                                         early_stopping_rounds=None,
                                         enable_categorical=False, eval_metric=None,
                                         feature_types=None, gamma=0, gpu_id=None,
                                         grow_policy=None, importance_type=None,
                                         interaction_constraints=None,
                                         learning_rate=0.1, max_bin=None,
                                         max_cat_threshold=None,
                                         max_cat_to_onehot=None,
                                         max_delta_step=None, max_depth=3,
                                         max_leaves=None, min_child_weight=1,
                                         missing=nan, monotone_constraints=None,
                                         n_estimators=1000, n_jobs=None, nthread=8,
                                         num_class=10, num_parallel_tree=None, ...),
                 param_grid={'reg_alpha': [0.01, 0.1, 1]}, scoring='roc_auc_ovr',
                 verbose=10)




```python
getTrainScores(gsearch2)
```




    ({0: "mean:0.8156626792866941params{'reg_alpha': 0.01}",
      1: "mean:0.8156025240054869params{'reg_alpha': 0.1}",
      2: "mean:0.8154581849108367params{'reg_alpha': 1}"},
     {'best_mean': 0.8156626792866941, 'best_param': {'reg_alpha': 0.01}})




```python
opt_reg_alpha = 0.01
```

## Section 3.3.1 Final Model Output & Loss Visualization (w/ Optimized `max_depth`, `min_child_weight`, & `reg_alpha`)

Below we output the model with the determined optimal `max_depth`, `min_child_height`, and `reg_alpha` found above. We evoke the functions defined in Section 3.0 to visually and numerically break down `mlogloss` and `merror`.


```python
xgb_clf3 = XGBClassifier(learning_rate=0.1,
                         n_estimators=1000,
                         max_depth=opt_max_depth, # tuned (previously)
                         min_child_weight=opt_min_child, # tuned (previously)
                         reg_alpha=opt_reg_alpha, # tuned
                         gamma=0,
                         subsample=0.8,
                         colsample_bytree=0.8,
                         objective='multi:softproba',
                         nthread=10,
                         num_class=10,
                         seed=1234567)

fitXgb(xgb_clf3, training_data)
```

    Fitting model...
    Fitting done!
    [0]	test-mlogloss:2.24656	test-merror:0.71280	train-mlogloss:2.24377	train-merror:0.69944
    

    /Users/sunnyson/opt/anaconda3/lib/python3.9/site-packages/xgboost/core.py:617: FutureWarning: Pass `evals` as keyword args.
      warnings.warn(msg, FutureWarning)
    

    [99]	test-mlogloss:1.71240	test-merror:0.67180	train-mlogloss:1.61757	train-merror:0.62124
    -- Model Report --
    XGBoost Accuracy: 0.3142
    XGBoost F1-Score (Micro): 0.3142
    XGBoost F1-Score (Macro): 0.3034744675861638
    


    
![png](images/Sunny%20Son%20IML%20Capstone_73_3.png)
    



    
![png](images/Sunny%20Son%20IML%20Capstone_73_4.png)
    


## Section 3.4 Results & Discussion

From the above, we can see that an `XGBoost` model initiated with the hyperparameters `max_depth=3`, `min_child_weight=1`, `reg_alpha=0.01` makes for the lowest `mlogloss=1.66697` and `merror=0.65900` with a `micro f1=0.345`, a `macro f1=0.33689`, and `accuracy=0.345`.

We can interpret this as meaning `XGBoost` as not having a too-distinctive ability to differentiate `music_genre` based on the first 3 Principal Components of the `musicData.csv` dataset. For a real-world dataset, achieveing `~0.25` is impressive enough, so the model is not without merit. However, further results (such as the AUC ROC plot showin the section below) will moreso elucidate predictive ability versus guessing. 

# Section 4.0 AUC ROC of Fitted & Tuned `XGBoost` Model

Below we define a function `plot_ROC_curve` that takes in the parameters `model`, `xtrain`, `ytrain`, `xtest`, and `ytest` to output the AUC ROC of each multiclass (multiple classes but each sample can only belong to one class) entry. Furthermore, different aggregations (`micro` and `macro` averages) of the AUC ROC curve of the model as a whole are presented.

This was done as per the spec sheet, to visually output not only the AUC ROC curve of every class, but also the AUC ROC performance of the model as a whole through different consolidation methods (`micro` and `macro`).


```python
def plot_ROC_curve(model, xtrain, ytrain, xtest, ytest):

    # Creating visualization with the readable labels
    visualizer = ROCAUC(model, encoder={0: 'Alternative', 
                                        1: 'Anime', 
                                        2: 'Blues',
                                        3: 'Classical',
                                        4: 'Country',
                                        5: 'Electronic',
                                        6: 'Hip-Hop',
                                        7: 'Jazz',
                                        8: 'Rap',
                                        9: 'Rock'})
                                        
    # Fitting to the training data first then scoring with the test data                                    
    visualizer.fit(xtrain, ytrain)
    visualizer.score(xtest, ytest)
    visualizer.show()
    
    return visualizer
```


```python
clf_choice = xgb_clf3
plot_ROC_curve(clf_choice, training_data['X_train'], y_train, training_data['X_test'], y_test)
```


    
![png](images/Sunny%20Son%20IML%20Capstone_77_0.png)
    





    ROCAUC(ax=<AxesSubplot:title={'center':'ROC Curves for XGBClassifier'}, xlabel='False Positive Rate', ylabel='True Positive Rate'>,
           encoder={0: 'Alternative', 1: 'Anime', 2: 'Blues', 3: 'Classical',
                    4: 'Country', 5: 'Electronic', 6: 'Hip-Hop', 7: 'Jazz',
                    8: 'Rap', 9: 'Rock'},
           estimator=XGBClassifier(base_score=None, booster=None, callbacks=None,
                                   colsample_bylevel=None, colsample_by...
                                   feature_types=None, gamma=0, gpu_id=None,
                                   grow_policy=None, importance_type=None,
                                   interaction_constraints=None, learning_rate=0.1,
                                   max_bin=None, max_cat_threshold=None,
                                   max_cat_to_onehot=None, max_delta_step=None,
                                   max_depth=3, max_leaves=None, min_child_weight=1,
                                   missing=nan, monotone_constraints=None,
                                   n_estimators=1000, n_jobs=None, nthread=10,
                                   num_class=10, num_parallel_tree=None, ...))



# Section 4.1 Results & Discussion

From the above analysis, we see that even given the low `accuracy` and `f1` metrics, we obtain above-random performance for the AUC ROC.

With a range of `min=(Alternative=0.71)` to `max=(Classical=0.96)`, the scores of individual features `ovr` (one versus rest comparison mentioned earlier) are displayed.

Also shown are the `micro=0.83` and `macro=0.81` averages of the AUC ROC curve of the entire model as a whole.

I interpret these numbers for the AUC ROC as showing that our model best classifies `Classical` music, and worstly does so for `Alternative`.

The `micro` and `macro` aggregations individually and combined show the efficacy of this model to perform better than random guessing.

From this, I think the most important factor of classification in this dataset is the availability of predictors that are able to better differentiate music (such as popularity).

# Section 5.0 Works Cited

[1] <span id='fn1'> https://ledgernote.com/columns/music-theory/circle-of-fifths-explained/ </span>

[2] <span id='fn2'> https://towardsdatascience.com/xgboost-for-multi-class-classification-799d96bcd368 </span>

[3] <span id='fn3'> https://medium.com/swlh/how-to-create-an-auc-roc-plot-for-a-multiclass-model-9e13838dd3de </span>

[4] <span id='fn4'> https://datascience.stackexchange.com/questions/15989/micro-average-vs-macro-average-performance-in-a-multiclass-classification-settin </span>

[5] <span id='fn5'> https://www.nvidia.com/en-us/glossary/data-science/xgboost/ </span>

# Section 6.0 Appendix

## Section 6.1 Depreciated Code


```python
df_drop = df.drop(index=df[(df['duration_ms'] == -1)].index, axis=0).reset_index(drop=True)
```


```python
string_data = ['key']
categorical_data = ['mode', 'music_genre']

smap = lambda x: re.search('\w+(?!_)', x)[0]
cmap = lambda x: re.search('\w+(?=_name)|mode|((?<=music_)\w+)', x)[0]
```


```python
encode_string = {}
for i, col in zip(range(len(string_data)), string_data):
    encode_string[f'encode_{smap(col)}'] = LabelEncoder().fit(df[[col]])
    
encode_categorical = {}
for i, col in zip(range(len(categorical_data)), categorical_data):
    encode_string[f'encode_{cmap(col)}'] = LabelEncoder().fit(df[[col]])
```


```python
df_process = pd.get_dummies(data=df_encode, prefix='mode', columns=['mode'])

df_process = df_process[['popularity', 'acousticness', 'danceability', 'duration_ms', 
                         'energy', 'instrumentalness', 'key', 'liveness', 'loudness', 'mode_Major', 'mode_Minor', 
                         'speechiness', 'tempo', 'valence', 'music_genre']]

df_process[['tempo', 'mode_Major', 'mode_Minor']] = df_process[['tempo', 'mode_Major', 'mode_Minor']].apply(pd.to_numeric)
```


```python
tts_dict = {}
for i in range(10):
    tts = train_test_split(df_encode.loc[(df_encode['music_genre'] == i), :'music_genre'], 
                           df_encode.loc[(df_encode['music_genre'] == i), 'music_genre'],
                           test_size=0.1)

    tts_dict[f'genre_{i}_tts'] = tts
```
