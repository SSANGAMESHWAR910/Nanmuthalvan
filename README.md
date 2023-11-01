# Nanmuthalvan
Phase 1
Drive link
https://docs.google.com/document/d/1-2iPDpMvqFML0FsqsgpimlnMDw1BxOcR/edit?usp=drivesdk&ouid=105866668187211126962&rtpof=true&sd=true


phase 2
Drive link
https://docs.google.com/document/d/1EA2SD6f4xQrDDq0kuUBRi4iK9vgtvxxG/edit?usp=drivesdk&ouid=105866668187211126962&rtpof=true&sd=true


Phase 3
Drive link
https://docs.google.com/document/d/1EnXZuJ-kTmEkusUkfJAnvCbrYZpFzlrU/edit?usp=drivesdk&ouid=105866668187211126962&rtpof=true&sd=true


drive link phase 4
https://docs.google.com/document/d/1FARov3rVVhhCVeyk7yv-BobBCakK3d-j/edit?usp=drivesdk&ouid=105866668187211126962&rtpof=true&sd=true 

drive link phase 5
https://docs.google.com/document/d/1G31jrShdMxURGnfnouD6HrCaXNQJASgb/edit?usp=drivesdk&ouid=105866668187211126962&rtpof=true&sd=true

Summary

IMDb is the world’s most popular and authoritative source for movie, TV, and celebrity content. IMDb users often look at ratings to get an idea of how good movies are, so that they can decide what movies to watch or which ones to prioritize. However, movies that are not yet released don’t have ratings, and even the ones with few votes often change as more users vote. Therefore, I wrote code to predict IMDb ratings of new movies based on various features, such as budget, actors, directors, writers, release year, genres, and plot. While others have used linear regressions to predict ratings of movies in general, those predictions rely on features like movie earnings or number of votes, which would not be available for new movies. I instead implemented two more algorithms to test the predictions and its accuracy.

Libraries
 Pandas
 Matplotlib
 Warnings 
 Seaborn

 Dataset link : https://www.kaggle.com/datasets/luiscorter/netflix-original-films-imdb-scores

 Program with Output 

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
dataset = pd.read_csv("../Dataset/IMDB_movie_reviews_details.csv")
dataset
dataset.info()

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1000 entries, 0 to 999
Data columns (total 10 columns):
 #   Column     Non-Null Count  Dtype  
---  ------     --------------  -----  
 0   id         1000 non-null   int64  
 1   name       1000 non-null   object 
 2   year       1000 non-null   object 
 3   runtime    1000 non-null   int64  
 4   genre      1000 non-null   object 
 5   rating     1000 non-null   float64
 6   meta score  841 non-null    float64
 7   timeline   1000 non-null   object 
 8   votes      1000 non-null   object 
 9   gross      829 non-null    object 
D types: float64(2), int64(2), object(6)
memory usage: 54.8+ KB
dataset.head()
dataset.isna().sum()

id                  0
name           0
year             0
runtime       0
genre           0
rating             0
meta score    159
timeline        0
votes             0
gross             171
d type:            int64
Deleting first column
df = dataset
df = df.drop(['id'], axis=1)
df.head()

Year column converted to int
df['year'] = df['year'].str.replace('I', '')
df['year'] = df['year'].str.replace(' ', '')
df['year'] = pd.to_numeric(df["year"])
df.info()

<class 'pandas.core.frame.DataFrame'>
Range Index: 1000 entries, 0 to 999
Data columns (total 9 columns):
 #   Column     Non-Null Count  D type  
---  ------     --------------  -----  
 0   name       1000 non-null   object 
 1   year       1000 non-null   int64  
 2   runtime    1000 non-null   int64  
 3   genre      1000 non-null   object 
 4   rating     1000 non-null   float64
 5   meta score  841 non-null    float64
 6   timeline   1000 non-null   object 
 7   votes      1000 non-null   object 
 8   gross      829 non-null    object 
d types: float64(2), int64(2), object(5)
memory usage: 50.8+ KB
Votes column converted to int
df['votes'] = df['votes'].str.replace(',', '')
df['votes'] = pd.to_numeric(df["votes"])
df.info()

<class 'pandas.core.frame.DataFrame'>
Range Index: 1000 entries, 0 to 999
Data columns (total 9 columns):
 #   Column     Non-Null Count  D type  
---  ------     --------------  -----  
 0   name       1000 non-null   object 
 1   year       1000 non-null   int64  
 2   runtime    1000 non-null   int64  
 3   genre      1000 non-null   object 
 4   rating     1000 non-null   float64
 5   meta score  841 non-null    float64
 6   timeline   1000 non-null   object 
 7   votes      1000 non-null   int64  
 8   gross      829 non-null    object 
D types: float64(2), int64(3), object(4)
memory usage: 54.8+ KB
Gross column converted to int
df['gross'] = df['gross'].str.replace('$', '')
df['gross'] = df['gross'].str.replace('M', '')
df['gross'] = pd.to_numeric(df["gross"])
df.info()

<class 'pandas.core.frame.DataFrame'>
Range Index: 1000 entries, 0 to 999
Data columns (total 9 columns):
 #   Column     Non-Null Count  D type  
---  ------     --------------  -----  
 0   name       1000 non-null   object 
 1   year       1000 non-null   int64  
 2   runtime    1000 non-null   int64  
 3   genre      1000 non-null   object 
 4   rating     1000 non-null   float64
 5   meta  score  841 non-null    float64
 6   timeline   1000 non-null   object 
 7   votes      1000 non-null   int64  
 8   gross      829 non-null    float64
D types: float64(3), int64(3), object(3)
memory usage: 58.7+ KB 
Checking how numbers correlate
sns.pairplot(df)
![image](https://github.com/HARIPRASATH250504/Nanmuthalvan/assets/146343467/0c5e592e-5de1-4b24-a117-34fc39f05355)

<seaborn.axisgrid.PairGrid at 0x336aeef8>
 ![image](https://github.com/HARIPRASATH250504/Nanmuthalvan/assets/146343467/834da68d-5f28-4008-bb35-391282d83c79)

sns.pairplot(df, hue='rating')

<seaborn.axisgrid.PairGrid at 0x3876cd18>
 

plt.figure()
cor = df.corr()
sns.heatmap(cor, annot=True, cmap='coolwarm')
plt.ylim()

(6.0, 0.0)
![image](https://github.com/HARIPRASATH250504/Nanmuthalvan/assets/146343467/f1b7f8b5-6bd6-480c-b948-f43d2bbb61de)

 Feature Engineering
Df
df[["genre_1","genre_2","genre_3"]] = df['genre'].str.split(',', n = 3, expand=True)
df = df.drop(['genre'], axis=1)
df['genre_1'] = df['genre_1'].str.replace(' ', '')
df['genre_2'] = df['genre_2'].str.replace(' ', '')
df['genre_3'] = df['genre_3'].str.replace(' ', '')
l1 = df.genre_1.unique()
l2 = df.genre_2.unique()
l3 = df.genre_3.unique()
l = list(l1) + list(l2) + list(l3)
l = [i for i in l if i]
l = list(set(l))
print(l)

['Action', 'History', 'Mystery', 'Sport', 'Romance', 'Western', 'Thriller', 'Musical', 'Animation', 'Film-Noir', 'Music', 'Horror', 'War', 'Sci-Fi', 'Comedy', 'Adventure', 'Fantasy', 'Crime', 'Family', 'Biography', 'Drama']
len(l)
21
listofzeros = [0] * 1000
for genre in l:
    df[genre] = listofzeros
df.head()
for genre in l:
    for x in range(1000):
        if df.at[x, 'genre_1'] == genre or df.at[x, 'genre_2'] == genre or df.at[x, 'genre_3'] == genre:
            df.at[x, genre] = 1
df.head()
Checking how genre correlate
plt.figure(figsize=(21,21))
cor = df[l].corr()
sns.heatmap(cor, annot=True, cmap='coolwarm')
plt.ylim()

(21.0, 0.0)
![image](https://github.com/HARIPRASATH250504/Nanmuthalvan/assets/146343467/abcade4f-ae14-4c3f-9af3-e05ad5d6d970)

 Removing unwanted columns for model training
df.info()

<class 'pandas.core.frame.DataFrame'>
Rang  Index: 1000 entries, 0 to 999
Data columns (total 32 columns):
 #   Column     Non-Null Count  D  type  
---  ------     --------------  -----  
 0   name       1000 non-null   object 
 1   year       1000 non-null   int64  
 2   runtime    1000 non-null   int64  
 3   rating     1000 non-null   float64
 4   meta  score  841 non-null    float64
 5   timeline   1000 non-null   object 
 6   votes      1000 non-null   int64  
 7   gross      829 non-null    float64
 8   genre_1    1000 non-null   object 
 9   genre_2    892 non-null    object 
 10  genre_3    643 non-null    object 
 11  Action     1000 non-null   int64  
 12  History    1000 non-null   int64  
 13  Mystery    1000 non-null   int64  
 14  Sport      1000 non-null   int64  
 15  Romance    1000 non-null   int64  
 16  Western    1000 non-null   int64  
 17  Thriller   1000 non-null   int64  
 18  Musical    1000 non-null   int64  
 19  Animation  1000 non-null   int64  
 20  Film-Noir  1000 non-null   int64  
 21  Music      1000 non-null   int64  
 22  Horror     1000 non-null   int64  
 23  War        1000 non-null   int64  
 24  Sci-Fi     1000 non-null   int64  
 25  Comedy     1000 non-null   int64  
 26  Adventure  1000 non-null   int64  
 27  Fantasy    1000 non-null   int64  
 28  Crime      1000 non-null   int64  
 29  Family     1000 non-null   int64  
 30  Biography  1000 non-null   int64  
 31  Drama      1000 non-null   int64  
D types: float64(3), int64(24), object(5)
memory usage: 230.5+ KB
#Not taking 'Horror' column to avoid dummy variable trap
df_model = df[['year', 'runtime', 'votes', 'metascore', 'gross', 'Mystery', 'Drama', 'Musical', 'Fantasy', 'Adventure', 'Western', 'Thriller', 'War', 'Biography', 'Family', 'Sport', 'Film-Noir', 'Music', 'Sci-Fi', 'Animation', 'Romance', 'Crime', 'Action', 'Comedy', 'History', 'rating']]
df_model.head()
df_model.info()

<class 'pandas.core.frame.DataFrame'>
Range  Index: 1000 entries, 0 to 999
Data columns (total 26 columns):
 #   Column     Non-Null Count  D type  
---  ------     --------------  -----  
 0   year       1000 non-null   int64  
 1   runtime    1000 non-null   int64  
 2   votes      1000 non-null   int64  
 3   meta score  841 non-null    float64
 4   gross      829 non-null    float64
 5   Mystery    1000 non-null   int64  
 6   Drama      1000 non-null   int64  
 7   Musical    1000 non-null   int64  
 8   Fantasy    1000 non-null   int64  
 9   Adventure  1000 non-null   int64  
 10  Western    1000 non-null   int64  
 11  Thriller   1000 non-null   int64  
 12  War        1000 non-null   int64  
 13  Biography  1000 non-null   int64  
 14  Family     1000 non-null   int64  
 15  Sport      1000 non-null   int64  
 16  Film-Noir  1000 non-null   int64  
 17  Music      1000 non-null   int64  
 18  Sci-Fi     1000 non-null   int64  
 19  Animation  1000 non-null   int64  
 20  Romance    1000 non-null   int64  
 21  Crime      1000 non-null   int64  
 22  Action     1000 non-null   int64  
 23  Comedy     1000 non-null   int64  
 24  History    1000 non-null   int64  
 25  rating     1000 non-null   float64
D types: float64(3), int64(23)
memory usage: 203.2 KB
Calculating mean and filling missing data
X = df_model.iloc[:,:-1].values
y = df_model.iloc[:,25].values
np.set_printoptions(suppress=True)
X

array([[   1994.,     142., 2394059., ...,       0.,       0.,       0.],
       [   1972.,     175., 1658439., ...,       0.,       0.,       0.],
       [   2020.,     153.,   78266., ...,       0.,       0.,       0.],
       ...,
       [   1953.,     118.,   37753., ...,       0.,       0.,       0.],
       [   1953.,     118.,   44086., ...,       0.,       0.,       0.],
       [   1944.,      97.,   26903., ...,       0.,       0.,       0.]])
X[0,:]

array([   1994.  ,     142.  , 2394059.  ,      80.  ,      28.34,
             0.  ,       1.  ,       0.  ,       0.  ,       0.  ,
             0.  ,       0.  ,       0.  ,       0.  ,       0.  ,
             0.  ,       0.  ,       0.  ,       0.  ,       0.  ,
             0.  ,       0.  ,       0.  ,       0.  ,       0.  ])
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
#X_train, X_test
#y_train, y_test

Implementing Algorithms
Linear Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)
accuracy = regressor.score(X_test,y_test)
print('Accuracy of the model is',accuracy*100,'%')

Accuracy of the model is 49.27193177432364 %
Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
for degree in range(1,4):
    poly_reg = PolynomialFeatures(degree = degree)
    X_poly = poly_reg.fit_transform(X)
    lin_reg2 = LinearRegression()
    lin_reg2.fit(X_poly,y)
    accuracy = lin_reg2.score(poly_reg.fit_transform(X),y)
    print('Accuracy of the model is with degree',str(degree),'=',accuracy*100,'%')

Accuracy of the model is with degree 1 = 44.74549675905853 %
Accuracy of the model is with degree 2 = 55.163495947730986 %
Accuracy of the model is with degree 3 = 69.56899235247278 %
Random Forest Regression
from sklearn.ensemble import RandomForestRegressor
x = 500
regressor = RandomForestRegressor(n_estimators = x)
regressor.fit(X,y)
accuracy = regressor.score(X,y)
print('Accuracy of the model with',str(x),'n_estimators','=',accuracy*100,'%')
Accuracy of the model with 500 n_estimators = 93.97201024913363 %
SVR
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y.reshape(-1,1))
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X,y)
accuracy = regressor.score(X,y)
print('Accuracy of the model is',accuracy*100,'%')

Accuracy of the model is 59.742573708014255 %



