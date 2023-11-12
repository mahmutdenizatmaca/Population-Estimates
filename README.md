# Population-Estimates
df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 740 entries, 0 to 739
Data columns (total 6 columns):
 #   Column           Non-Null Count  Dtype  
---  ------           --------------  -----  
 0   STATISTIC Label  740 non-null    object 
 1   Year             740 non-null    int64  
 2   Age Group        740 non-null    object 
 3   Sex              740 non-null    object 
 4   UNIT             740 non-null    object 
 5   VALUE            740 non-null    float64
dtypes: float64(1), int64(1), object(4)
memory usage: 34.8+ KB
df["Age Group"].unique()
array(['0 - 14 years', '15 - 24 years', '25 - 44 years', '45 - 64 years',
       '65 years and over'], dtype=object)
df.isnull().values.any()
False
X = df.iloc[:,1:4].values  
y = df.iloc[:,-1].values 
le=LabelEncoder()
df["Age Group"] = le.fit_transform(df["Age Group"])
df["Sex"] = le.fit_transform(df["Sex"])
df["Year"] = le.fit_transform(df["Year"])
df
STATISTIC Label	Year	Age Group	Sex	UNIT	VALUE
0	Population Estimates (Persons in April)	0	0	1	Thousand	434.6
1	Population Estimates (Persons in April)	0	0	0	Thousand	416.6
2	Population Estimates (Persons in April)	0	1	1	Thousand	234.9
3	Population Estimates (Persons in April)	0	1	0	Thousand	217.7
4	Population Estimates (Persons in April)	0	2	1	Thousand	393.4
...	...	...	...	...	...	...
735	Population Estimates (Persons in April)	73	2	0	Thousand	749.6
736	Population Estimates (Persons in April)	73	3	1	Thousand	661.5
737	Population Estimates (Persons in April)	73	3	0	Thousand	677.2
738	Population Estimates (Persons in April)	73	4	1	Thousand	379.9
739	Population Estimates (Persons in April)	73	4	0	Thousand	426.4
740 rows × 6 columns

import matplotlib.pyplot as plt 
import seaborn as sb
corr_= df.corr()
plt.figure(figsize=(15,10))
sb.heatmap(corr_,annot=True)
plt.show()

labelencoder_X=LabelEncoder()
X[:,-2] = labelencoder_X.fit_transform(X[:,-2])
print(X)
[[1950 0 'Male']
 [1950 0 'Female']
 [1950 1 'Male']
 ...
 [2023 3 'Female']
 [2023 4 'Male']
 [2023 4 'Female']]
labelencoder_X=LabelEncoder()
X[:,-1] = labelencoder_X.fit_transform(X[:,-1])
print(X)
[[1950 0 1]
 [1950 0 0]
 [1950 1 1]
 ...
 [2023 3 0]
 [2023 4 1]
 [2023 4 0]]
labelencoder_X=LabelEncoder()
X[:,-3] = labelencoder_X.fit_transform(X[:,-3])
print(X)
[[0 0 1]
 [0 0 0]
 [0 1 1]
 ...
 [73 3 0]
 [73 4 1]
 [73 4 0]]
np.unique(X[:,-2])
​
array([0, 1, 2, 3, 4], dtype=object)
np.unique(X[:,-1])
array([0, 1], dtype=object)
np.unique(X[:,-3])
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
       19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
       36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
       53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
       70, 71, 72, 73], dtype=object)
df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 740 entries, 0 to 739
Data columns (total 6 columns):
 #   Column           Non-Null Count  Dtype  
---  ------           --------------  -----  
 0   STATISTIC Label  740 non-null    object 
 1   Year             740 non-null    int64  
 2   Age Group        740 non-null    int32  
 3   Sex              740 non-null    int32  
 4   UNIT             740 non-null    object 
 5   VALUE            740 non-null    float64
dtypes: float64(1), int32(2), int64(1), object(2)
memory usage: 29.0+ KB
import seaborn as sns
import matplotlib.pyplot as plt   
%matplotlib inline
sns.boxplot(x = "Age Group", data = df)
<AxesSubplot:xlabel='Age Group'>

sns.boxplot(x = "Year", data = df)
<AxesSubplot:xlabel='Year'>

import matplotlib.pyplot as plt 
plt.bar(df["Age Group"],df["VALUE"])
plt.xlabel('Age Group')
plt.ylabel("Values")
​
plt.show()

plt.bar(df["Sex"],df["VALUE"])
plt.xlabel('Sex')
plt.ylabel("Values")
​
plt.show()

sns.distplot(df['VALUE'])
C:\ProgramData\Anaconda3\lib\site-packages\seaborn\distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
  warnings.warn(msg, FutureWarning)
<AxesSubplot:xlabel='VALUE', ylabel='Density'>

df.corr()
Year	Age Group	Sex	VALUE
Year	1.000000e+00	3.820731e-16	4.720047e-16	0.482780
Age Group	3.820731e-16	1.000000e+00	1.612527e-17	-0.411442
Sex	4.720047e-16	1.612527e-17	1.000000e+00	-0.002944
VALUE	4.827801e-01	-4.114422e-01	-2.944209e-03	1.000000
print(X.shape)
print(type(X))
(740, 3)
<class 'numpy.ndarray'>
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4)
LinearRegression
from sklearn.linear_model import LinearRegression
ln = LinearRegression()
ln.fit(X_train,y_train)
LinearRegression()
predictions = ln.predict(X_test)
sns.scatterplot(y_test,predictions)
C:\ProgramData\Anaconda3\lib\site-packages\seaborn\_decorators.py:36: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  warnings.warn(
<AxesSubplot:>

ln.score(X_test,y_test)
0.4028761714544208
