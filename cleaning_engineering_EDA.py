#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 15:58:07 2020

@author: yiannimercer
"""


##Import Libraries
import os 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

##Explore data
HC_desc = pd.read_csv('HomeCredit_columns_description.csv', encoding='latin')

HC_desc.head()

app_train = pd.read_csv('application_train.csv')

app_train.head()

app_train.shape

##Training Data

#we can see the TARGET column, which takes on either a 1 or 0, is in this dataset. This is the label we will try to predict in our machine learnign model
app_train.columns
app_train.TARGET

app_train['TARGET'].plot.hist(bins = app_train['TARGET'].value_counts().shape[0]+1,alpha = 0.75)

##Testing data features
app_test = pd.read_csv('application_test.csv')
app_test.shape

app_test.head()


##Data Cleaning

# Function to calculate missing values by column
def missing_values_table(df):
    #total missing values 
    miss_val = df.isnull().sum()
    
    #Total values
    tot_val = df.count() 
    
    #Percentage of missing values
    miss_val_percent = 100 * df.isnull().sum() / len(df)
        
    #Make a table with the results
    miss_val_table = pd.concat([miss_val, tot_val, miss_val_percent], axis=1)
    
    #Rename columns
    miss_val_table_ren_columns = miss_val_table.rename(
        columns = {0:'Missing Values',1:'Valid Values',2:'% of Total Values'})
    
    #Sort the table by percenage of missing descending 
    miss_val_table_ren_columns = miss_val_table_ren_columns[
        miss_val_table_ren_columns.iloc[:,2] != 0].sort_values(
            '% of Total Values', ascending = False).round(1)
        
    # Print some summary information
    print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
          "There are " + str(miss_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")
    
    #Return the dataframe with missing information
    return miss_val_table_ren_columns
            
missing_values = missing_values_table(app_train)
missing_values.head(20)

#Categorical Variable Encoding

app_train.dtypes.value_counts()

app_train.select_dtypes('object').apply(pd.Series.nunique, axis = 0)

#sklearn preprocessing for dealing with categorical variables


# new label encoder object 
le = LabelEncoder()
le_count = 0 

#Iterate by object columns
for col in app_train:
    if app_train[col].dtype == 'object':
        # category with only two options
        if len(list(app_train[col].unique())) <= 2:
            #train on the column
            le.fit(app_train[col])
            
            #transform both training & test data
            # - note that htis applies to the case where training data has equal or more categorical entries than the test data
            app_train[col] = le.transform(app_train[col])
            app_test[col] = le.transform(app_test[col])
            
            #count the number of 2 category objects
            le_count += 1
            
print("%d columns were label encoded." % le_count)

#one-hot encoding of categorical variables 
app_train = pd.get_dummies(app_train)
app_test = pd.get_dummies(app_test)

print('Training Features shape: ', app_train.shape)
print('Testing Features shape: ', app_test.shape)
      

      
train_labels = app_train['TARGET']

#align
app_train, app_test = app_train.align(app_test, join = 'inner',axis =1)

# put back the TARGFET column - creating a new column
app_train['TARGET'] = train_labels

print('Training Features shape: ', app_train.shape)
print('Testing Features shape: ', app_test.shape)

#Outliers and anomalies

#age
print((app_train['DAYS_BIRTH']/-365).describe() )
print('\nAdults, between 20 and 69 - looks reasonable')

(app_train['DAYS_BIRTH']/-365).hist(bins=20)
    
#Years employed

print(  (app_train['DAYS_EMPLOYED']).describe() , '\n')

print(  (app_train['DAYS_EMPLOYED']/365).describe()  )

print('\nAnomalous!')        
            
app_train['DAYS_EMPLOYED'].hist(bins=55)           

#create an anomalous flag column
app_train['DAYS_EMPLOYED_ANOM'] = app_train['DAYS_EMPLOYED'] == 365243            

#replace the anomalous flag column with nan
app_train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)            

(app_train['DAYS_EMPLOYED']/-365).hist(bins=20)
plt.title('DAYS_EMPLOYED Histogram')
plt.xlabel('Years Employed')
plt.ylabel('Years Employed')        

#same change to test data
app_test['DAYS_EMPLOYED'].hist(bins=55)        
 
#create an anomalous flag column
app_test['DAYS_EMPLOYED_ANOM'] = app_test['DAYS_EMPLOYED'] == 365243  

# Replace the anomalous values with nan
app_test['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)

(app_test['DAYS_EMPLOYED']/-365).hist(bins=20)
plt.title('DAYS_EMPLOYED Histogram')
plt.xlabel('Years Employed')
plt.ylabel('Years Employed')

print('There are %d anomalies in the test data out of %d entries' % (app_test["DAYS_EMPLOYED_ANOM"].sum(), len(app_test)))

##Corelations

correlation = app_train.corr()['TARGET'].sort_values()
print('Most Positive Correlations:\n', correlation.tail(15))
print('\nMost Negative Correlations:\n', correlation.head(15))

#visualize the effect of age on TARGET with kde plot using sns
plt.style.use('fivethirtyeight')
plt.figure(figsize = (10,8))
#KDE plot of loans that were repaid on time
sns.kdeplot(abs(app_train.loc[app_train['TARGET'] == 0, 'DAYS_BIRTH']/365), label = 'target == 0 (repaid)',linewidth=3)
#KDE plot of loans which were not repaid on time
sns.kdeplot(  abs(app_train.loc[app_train['TARGET'] == 1, 'DAYS_BIRTH'] / 365), label = 'target == 1 (defaulted)', linewidth=3)
#Labeling of plot 
plt.xlabel("Age (Years)",fontsize = 14); plt.ylabel('Density',fontsize=14);plt.title("Distribution of Ages",fontsize =14)

#Age information into a separate df
age_data = app_train[['TARGET' ,'DAYS_BIRTH']]
age_data['YEARS_BIRTH'] = age_data['DAYS_BIRTH']/-365

#Bin the age data
age_data['YEARS_BINNED'] = pd.cut(age_data['YEARS_BIRTH'], bins = np.linspace(20,70, num =11))
age_data.head(20)

#Grouped by the bin and calculate the averages 
age_groups = age_data.groupby('YEARS_BINNED').mean()
print(age_groups)

#Graph the age bins and the average of the target as a bar plot
plt.bar(age_groups.index.astype(str), 100 * age_groups['TARGET'])
#Plot labeling
plt.xticks(rotation = 75);plt.xlabel('Age Group (Years)');plt.ylabel('Failure to Repay (%)')
plt.title('Failure to Repay by Age Groups')

##Extract the EXT_SOURCE variables and show correlations
ext_data = app_train[['TARGET', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]
ext_data_corrs = ext_data.corr()
print(ext_data_corrs)

#Heatmap of correlations  
plt.figure(figsize=(8,6))
sns.heatmap(ext_data_corrs, cmap = plt.cm.RdYlBu_r,vmin = -0.25,annot = True, vmax = 0.6)
plt.title('Correlation Heatmap')

#Iterate through the sources and plot 
for i, source in enumerate(['EXT_SOURCE_1', 'EXT_SOURCE_2','EXT_SOURCE_3']):
    #Create nwe subplot for each source
    plt.figure(figsize=(8,12))
    sns.kdeplot(app_train.loc[app_train['TARGET'] == 0, source], label = 'target == 0')
    # plot loans that were not repaid
    sns.kdeplot(app_train.loc[app_train['TARGET'] == 1, source], label = 'target == 1')
    
    # Label the plots
    plt.title('Distribution of %s by Target Value' % source)
    plt.xlabel('%s' % source); plt.ylabel('Density');
    
plt.tight_layout(h_pad = 2.5)

# Copy the data for plotting
plot_data = ext_data.drop(columns = ['DAYS_BIRTH']).copy()
# Add in the age of the client in years
plot_data['YEARS_BIRTH'] = age_data['YEARS_BIRTH']
# Drop na values and limit to first 10000 rows
plot_data = plot_data.dropna().loc[:10000, :]

# Function to calculate correlation coefficient between two columns
def corr_func(x,y,**kwargs):
    r = np.corrcoef(x,y)[0][1]
    ax = plt.gca()
    ax.annotate("r = {:.2f}".format(r),
                xy = (.2,.8), xycoords=ax.transAxes,
                size = 20)
#Create the pairgrid object
grid = sns.PairGrid(data = plot_data, size =3, diag_sharey=False,
                    hue = 'TARGET',
                    vars = [x for x in list(plot_data.columns) if x != 'TARGET'])

#Upper is a scatter plot 
grid.map_upper(plt.scatter, alpha = 0.2)

#Diagnol is a histogram 
grid.map_diag(sns.kdeplot)

#Bottom is a density plot 
grid.map_lower(sns.kdeplot, cmap = plt.cm.OrRd_r);

plt.suptitle('Ext Source and Age Features Pairs Plot', size = 32, y = 1.05);
    

##Feature Engineering

#Make a new dataframe for polynomial features
poly_features = app_train[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3','DAYS_BIRTH','TARGET']]
poly_features_test = app_test[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3','DAYS_BIRTH']]

#Imputer for handling missing values
imputer = SimpleImputer(strategy='median')

poly_target = poly_features['TARGET']

poly_features = poly_features.drop(columns = ['TARGET'])

#Impute missing values
poly_features = imputer.fit_transform(poly_features)
poly_features_test = imputer.transform(poly_features_test)
                                
#Create the polynomial object with specified degree
poly_transformer = PolynomialFeatures(degree = 3)

#Train the polynomial features
poly_transformer.fit(poly_features)

#Transform the features 
poly_features = poly_transformer.transform(poly_features)
poly_features_test = poly_transformer.transform(poly_features_test) 
print('Polynomial Features shape: ', poly_features.shape)

poly_transformer.get_feature_names(input_features= ['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3','DAYS_BIRTH'])[:27]

#Create a df of the train features
poly_features = pd.DataFrame(poly_features, 
                             columns = poly_transformer.get_feature_names((['EXT_SOURCE_1','EXT_SOURCE_2',
                                                                            'EXT_SOURCE_3','DAYS_BIRTH']))
                             )

#Add in target variable 
poly_features['TARGET'] = poly_target

#Find correlations with target variable 
poly_corrs = poly_features.corr()['TARGET'].sort_values()

# Display most negative and most positive
print(poly_corrs.head(10), '\n')
print(poly_corrs.tail(10))

#Create a df of the test features 
poly_features_test = pd.DataFrame(poly_features_test, 
                                  columns = poly_transformer.get_feature_names(['EXT_SOURCE_1', 'EXT_SOURCE_2', 
                                                                                'EXT_SOURCE_3', 'DAYS_BIRTH']))

#Merge polynomial features into training df
poly_features['SK_ID_CURR'] = app_train['SK_ID_CURR']
app_train_poly = app_train.merge(poly_features, on = 'SK_ID_CURR',how = 'left')

#Merge polynomial features into testing df
poly_features_test['SK_ID_CURR'] = app_test['SK_ID_CURR']
app_test_poly = app_test.merge(poly_features_test, on = 'SK_ID_CURR', how = 'left')

#Align the df's
app_train_poly, app_test_poly = app_train_poly.align(app_test_poly, join = 'inner',axis = 1)

# Print out the new shapes
print('Training data with polynomial features shape: ', app_train_poly.shape)
print('Testing data with polynomial features shape:  ', app_test_poly.shape)

#Some Domain Knowledge Feature Engineering

app_train_domain = app_train.copy()
app_test_domain = app_test.copy()

#CREDIT_INCOME_PERCENT: the percentage of the credit amount relative to a client's income
app_train_domain['CREDIT_INCOME_PERCENT'] = app_train_domain['AMT_CREDIT'] / app_train_domain['AMT_INCOME_TOTAL']

#ANNUITY_INCOME_PERCENT: the percentage of the loan annuity relative to a client's income
app_train_domain['ANNUITY_INCOME_PERCENT'] = app_train_domain['AMT_ANNUITY'] / app_train_domain['AMT_INCOME_TOTAL']

#CREDIT_TERM: the length of the payment in months (since the annuity is the monthly amount due
app_train_domain['CREDIT_TERM'] = app_train_domain['AMT_ANNUITY'] / app_train_domain['AMT_CREDIT']

#DAYS_EMPLOYED_PERCENT: the percentage of the days employed relative to the client's age
app_train_domain['DAYS_EMPLOYED_PERCENT'] = app_train_domain['DAYS_EMPLOYED'] / app_train_domain['DAYS_BIRTH']

#Do the same for the test data
app_test_domain['CREDIT_INCOME_PERCENT'] = app_test_domain['AMT_CREDIT'] / app_test_domain['AMT_INCOME_TOTAL']
app_test_domain['ANNUITY_INCOME_PERCENT'] = app_test_domain['AMT_ANNUITY'] / app_test_domain['AMT_INCOME_TOTAL']
app_test_domain['CREDIT_TERM'] = app_test_domain['AMT_ANNUITY'] / app_test_domain['AMT_CREDIT']
app_test_domain['DAYS_EMPLOYED_PERCENT'] = app_test_domain['DAYS_EMPLOYED'] / app_test_domain['DAYS_BIRTH']

#Visualizing the new variables

#iterate through the new features 
plt.figure(figsize = (15,20))

for i, feature in enumerate(['CREDIT_INCOME_PERCENT', 'ANNUITY_INCOME_PERCENT', 'CREDIT_TERM', 'DAYS_EMPLOYED_PERCENT']):
    #create new subplot for each feature
    plt.subplot(4,1,i+1)
    #plot repaid loans
    sns.kdeplot(app_train_domain.loc[app_train_domain['TARGET'] == 0, feature], label = 'target == 0')
    #plot loans that were not repaid
    sns.kdeplot(app_train_domain.loc[app_train_domain['TARGET'] == 1, feature], label = 'target == 1')
    #label plots
    plt.title('Distribution of the %s by Target Value' % feature)
    plt.xlabel('%s' % feature);plt.ylabel('Density')
    
plt.tight_layout(h_pad = 2.5)








