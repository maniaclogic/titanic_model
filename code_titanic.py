import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('train.csv')

dfi = df.set_index('PassengerId')
dfi.shape
dfi.columns
dfi.describe()
dfi.info()

#Write a program that calculates the number of surviving passengers and prints it to the screen.

surv = dfi[dfi['Survived'] == 1].count()
surv.head(1)

firstc = dfi[['Pclass', 'Survived']]
firstc_t = firstc[firstc['Pclass'] == 1]

t = firstc_t['Pclass'].value_counts()
f = firstc_t[firstc_t['Survived'] == 1]

perc = f['Survived'].value_counts() * 100 / t
perc

firstc_t.groupby('Pclass')['Survived'].value_counts().plot.pie()
plt.xlabel("63 % of first class passengers survived")

dfi['Age'].mean()

del dfi['Embarked']
del dfi['Cabin']
dfi1 = dfi.dropna()

dfi1.groupby('Survived')['Age'].mean()

dfi.reset_index(inplace=True)
dfi.set_index(['Survived'], inplace=True)
dfi['Age'].fillna({0:30.6, 1:28.3}, inplace=True)

dfi.reset_index(inplace=True)
dfi.groupby(['Survived', 'Sex'])['Pclass'].value_counts().unstack().plot.bar()
plt.ylabel("Person count")
plt.title("surviving/dead passengers separated by class and gender")

# Feature Engineering

#* normalizing --	de-mean, scale or otherwise transform features

#* scaling 	-- shift the mean and standard deviation

#* imputation 	-- fill missing values

#* one-hot encoding 	-- convert categories to binary columns

#* add features -- 	add extra polynomial or combined features

#* feature selection --	decide which features to use

dfi.set_index('PassengerId', inplace=True)

#Columns = Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare'

#binary (hot coding) - Sex, Pclass, SibSP, Parch

#normalizing - Fare, Age

dfi['Sex_bin'] = [1 if x =='male' else 0 for x in df['Sex']]

del dfi['Sex']
del dfi['Ticket']

Pc = dfi['Pclass']
Pclass_bin = pd.get_dummies(Pc)
Pclass_bin.columns = ['Pclass1', 'Pclass2', 'Pclass3']
dfi = pd.concat([dfi, Pclass_bin], axis=1)
del dfi['Pclass']

dfi['Age'] = dfi['Age'].apply(lambda x: x / 26501.77)
dfi['Fare'] = dfi['Fare'].apply(lambda x: x / 28693.9493)

Parch = dfi['Parch'].values
SibSp = dfi['SibSp'].values

dfi['family'] = (list(zip(Parch, SibSp)))
dfi['family'] =

#dfi['family'] = pd.DataFrame(Parch, SibSp)
#m = PolynomialFeatures(interaction_only=True)
#m.fit_transform(dfi['family'])

# Logistic Regression

#* sigmoid function --	function rapidly changing from 0 to 1
#* coefficients --	model parameters in the linear part
#* log probability --	result of the logistic function
#* threshold value --	probability at which a positive prediction is made (default 0.5)
#* log loss --	error function to be optimized
#* one-vs-rest --	strategy for multinomial regression
#* softmax --	error function for multinomial regression
