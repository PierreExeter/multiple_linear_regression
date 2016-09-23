import pandas as pd
import csv as csv
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew

#from sklearn.linear_model import LogisticRegression
#from sklearn.svm import SVC
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.cross_validation import KFold   #For K-fold cross validation
#from sklearn.tree import DecisionTreeClassifier, export_graphviz
#from sklearn import metrics

from sklearn.cross_validation import cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn import linear_model

# LOAD DATA INTO DATAFRAME
train_df = pd.read_csv("input/train.csv", header=0)
test_df  = pd.read_csv("input/test.csv", header=0)

# Explore data
print train_df.info()
print test_df.info()

train_column = list(train_df.columns.values)
test_column = list(test_df.columns.values)

# fill missing values

def fill_mode(df, column):
    """
    fill NaN values in column in df by the most common entry
    """
    
    mode_value = df[column].dropna().mode().values 
    
    if len(df[column][df[column].isnull()]) > 0:    
        print 'replace by mode: ', mode_value
        df.loc[df[column].isnull(), column] = mode_value 


def fill_mean(df, column):    
    """
    fill NaN values in column in df by the mean of the column
    """   

    mean_value = df[column].dropna().mean()

    if len(df[column][df[column].isnull()]) > 0:
        print 'replace by mean: ', mean_value
        df.loc[df[column].isnull(), column] = mean_value


def fill_median(df, column):    
    """
    fill NaN values in column in df by the median of the column
    """   

    median_value = df[column].dropna().median()

    if len(df[column][df[column].isnull()]) > 0:
        print 'replace by median: ', median_value
        df.loc[df[column].isnull(), column] = median_value


def outlier_mean(df, column, x):    
    """
    fill outlier values in column in df which are > x by the mean of the column
    """   

    mean_value = df[column][df[column] < x].mean()
    print 'replace by mean: ', mean_value
    df.loc[df[column] > x, column] = mean_value
    

def plot_distribution(column_number):
    """ plot histogram of variable column_number for train and test data """

    if train_df[train_column[column_number]].dtypes == object:
        # for categorical variables
        train_df[train_column[column_number]].value_counts().plot(kind='bar')
        plt.title('train data')
        plt.xlabel(train_column[column_number])
        plt.show()
        
        test_df[test_column[column_number]].value_counts().plot(kind='bar')
        plt.title('test data')
        plt.xlabel(test_column[column_number])        
        plt.show()
        
    else:
        # for quantitative variables
        train_df[train_column[column_number]].hist(bins=50)
        plt.title('train data')
        plt.xlabel(train_column[column_number])        
        
        plt.show()
        test_df[test_column[column_number]].hist(bins=50)
        plt.title('test data')
        plt.xlabel(test_column[column_number]) 
        plt.show()

def print_nb_null(column_number):
    """ print number of null observation in column_number for train and test data """
    print train_column[column_number]
    print train_df[train_column[column_number]].dtypes
    print 'nb of NaN in train data ', train_df[train_column[column_number]].isnull().sum()
    print 'nb of NaN in test data ', test_df[test_column[column_number]].isnull().sum()


def fill_no(column):
    
    """ fill NaN by 'NO' """
    
    train_df[train_column[column]] = train_df[train_column[column]].fillna('NO')
    test_df[test_column[column]] = test_df[test_column[column]].fillna('NO')
    

# MSZoning: few missing + categorical variable: replace by mode 
fill_mode(test_df, test_column[2])

# LotFrontage: outlier > 200
outlier_mean(train_df, train_column[3], 200)

# LotFrontage: missing values replace by mean
fill_mean(train_df, train_column[3])
fill_mean(test_df, test_column[3])

# Alley: NaN mean no alley so replace NaN by 'No'
fill_no(6)

# Utilities: few missing + categorical variable: replace by mode
fill_mode(test_df, test_column[9])

# Exterior1st: few missing + categorical variable: replace by mode
fill_mode(test_df, test_column[23])

# Exterior2nd: few missing + categorical variable: replace by mode
fill_mode(test_df, test_column[24])

fill_mode(train_df, train_column[25])
fill_mode(test_df, test_column[25])

fill_mode(train_df, train_column[26])
fill_mode(test_df, test_column[26])

fill_no(30)
fill_no(31)
fill_no(32)
fill_no(33)

fill_mode(test_df, test_column[34])

fill_no(35)

fill_mode(test_df, test_column[36])

fill_mode(test_df, test_column[37])

fill_mode(test_df, test_column[38])

fill_mode(train_df, train_column[42])

fill_mode(test_df, test_column[47])

fill_mode(test_df, test_column[48])

fill_mode(test_df, test_column[53])

fill_mode(test_df, test_column[55])

fill_no(57)
fill_no(58)

outlier_mean(test_df, test_column[59], 2050)

fill_mode(train_df, train_column[59])
fill_mode(test_df, test_column[59])

fill_no(60)

fill_mode(test_df, test_column[61])

fill_median(test_df, test_column[62])

fill_no(63)
fill_no(64)
fill_no(72)
fill_no(73)
fill_no(74)

fill_mode(test_df, test_column[78])

#column_number = 30
#plot_distribution(column_number)
#print_nb_null(column_number)


all_data = pd.concat((train_df.loc[:,'MSSubClass':'SaleCondition'],
                      test_df.loc[:,'MSSubClass':'SaleCondition']))
             
# deal with skewness

plt.rcParams['figure.figsize'] = (12.0, 6.0)
prices = pd.DataFrame({"price":train_df["SalePrice"], "log(price + 1)":np.log1p(train_df["SalePrice"])})
prices.hist()
plt.show()

#log transform the target:
train_df["SalePrice"] = np.log1p(train_df["SalePrice"])

#log transform skewed numeric features:
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

#compute skewness
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())) 
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index
all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

## create dummy variables to convert categorical variables with more than 2 levels 
## into dichotomic categorical variables (0 or 1): in order to run the linear regression
all_data = pd.get_dummies(all_data)

print all_data.info()

#creating matrices for sklearn:
X_train = all_data[:train_df.shape[0]].values
X_test = all_data[train_df.shape[0]:].values
y = train_df.SalePrice.values
idx = test_df['Id'].values


# linear regression

def linear_regression(x_train, y_train, x_test):
    """ linear regression model, return the prediction """
        
    # Create linear regression object
    linear = linear_model.LinearRegression()
        
    # Train the model using the training sets and check score
    linear.fit(x_train, y_train)
    print 'score: ', linear.score(x_train, y_train)
    
    #Equation coefficient and Intercept
    print('Coefficient: \n', linear.coef_)
    print('Intercept: \n', linear.intercept_)
    
    # cross validation prediction
    predicted = cross_val_predict(linear, x_train, y_train, cv=10)
    plt.scatter(y_train, predicted)
    plt.xlabel('Measured')
    plt.ylabel('Predicted')
    plt.show()
    
    #Predict Output
    return linear.predict(x_test)
    
out = linear_regression(X_train, y, X_test)

# convert back from log
out = np.expm1(out)

def write_output(filename, idx, out):
    """
    write model result to filename: 1st column is idx and 
    2nd column is out
    """        
    predictions_file = open(filename, 'wb')
    open_file_object = csv.writer(predictions_file)
    open_file_object.writerow(['Id', 'SalePrice'])
    open_file_object.writerows(zip(idx, out))
    predictions_file.close() 

write_output("output/log_linear.csv", idx, out)