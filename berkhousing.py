import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn import linear_model as lm

# Load data into dataframes from csv files
training_data = pd.read_csv('data/ames_train.csv')
test_data = pd.read_csv('data/ames_test.csv')

# 2000 observations and 82 features in training data
assert training_data.shape == (2000, 82)
# 930 observations and 81 features in test data
assert test_data.shape == (930, 81)
# SalePrice is hidden in the test data
assert 'SalePrice' not in test_data.columns.values
# Every other column in the test data should be in the training data
assert len(np.intersect1d(test_data.columns.values, 
                          training_data.columns.values)) == 81

def remove_outliers(data, variable, lower=-np.inf, upper=np.inf):
    '''
    Input:
      data (data frame): the table to be filtered
      variable (string): the column with numerical outliers
      lower (numeric): observations with values lower than or equal to this will be removed
      upper (numeric): observations with values higher than or equal to this will be removed
    
    Output:
      a winsorized data frame with outliers removed
      
    Note: This function should not change mutate the contents of data.
    ''' 
    return data.loc[(data[variable] > lower) & (data[variable] < upper), :]

def add_total_bathrooms(data):
    '''
    Input:
      data (data frame): a data frame containing at least 4 numeric columns 
            Bsmt_Full_Bath, Full_Bath, Bsmt_Half_Bath, and Half_Bath
    '''
    with_bathrooms = data.copy()
    bath_vars = ['Bsmt_Full_Bath', 'Full_Bath', 'Bsmt_Half_Bath', 'Half_Bath']
    weights = pd.Series([1, 1, 0.5, 0.5], index=bath_vars)
    with_bathrooms = with_bathrooms.fillna(0)
    with_bathrooms['TotalBathrooms'] = with_bathrooms['Bsmt_Full_Bath'] + with_bathrooms['Full_Bath'] + 1/2*(with_bathrooms['Bsmt_Half_Bath'] + with_bathrooms['Half_Bath'])
    return with_bathrooms

def find_rich_neighborhoods(data, n=3, metric=np.median):
    '''
    Input:
      data (data frame): should contain at least a string-valued Neighborhood
        and a numeric SalePrice column
      n (int): the number of top values desired
      metric (function): function used for aggregating the data in each neighborhood.
        for example, np.median for median prices
    
    Output:
      a list of the top n richest neighborhoods as measured by the metric function
    '''
    neighborhoods = data.groupby('Neighborhood').agg({'SalePrice':metric}).sort_values(by='SalePrice', ascending=False).head(n).index.tolist()
    return neighborhoods

def add_in_rich_neighborhood(data, neighborhoods):
    '''
    Input:
      data (data frame): a data frame containing a 'Neighborhood' column with values
        found in the codebook
      neighborhoods (list of strings): strings should be the names of neighborhoods
        pre-identified as rich
    Output:
      data frame identical to the input with the addition of a binary
      in_rich_neighborhood column
    '''
    data_copy = data.copy()
    data_copy['in_rich_neighborhood'] = data_copy['Neighborhood'].isin(neighborhoods).astype(int)
    return data_copy

def fix_fireplace_qu(data):
    '''
    Input:
      data (data frame): a data frame containing a Fireplace_Qu column.  Its values
                         should be limited to those found in the codebook
    Output:
      data frame identical to the input except with a refactored Fireplace_Qu column
    '''
    new_data = data.copy()
    new_data['Fireplace_Qu'] = data['Fireplace_Qu'].replace(to_replace=['NA', 'TA', 'Ex', 'Gd', 'Fa', 'Po'], value=['No Fireplace', 'Average', 'Excellent', 'Good', 'Fair', 'Poor']).fillna('No Fireplace')
    return new_data
    
def ohe_fireplace_qu(data):
    '''
    One-hot-encodes fireplace quality.  New columns are of the form Fireplace_Qu=QUALITY
    '''
    vec_enc = DictVectorizer()
    vec_enc.fit(data[['Fireplace_Qu']].to_dict(orient='records'))
    fireplace_qu_data = vec_enc.transform(data[['Fireplace_Qu']].to_dict(orient='records')).toarray()
    fireplace_qu_cats = vec_enc.get_feature_names()
    fireplace_qu = pd.DataFrame(fireplace_qu_data, columns=fireplace_qu_cats)
    data = pd.concat([data, fireplace_qu], axis=1)
    data = data.drop(columns=fireplace_qu_cats[0])
    return data

def rmse(actual, predicted):
    '''
    Calculates RMSE from actual and predicted values
    Input:
      actual (1D array): vector of actual values
      predicted (1D array): vector of predicted/fitted values
    Output:
      a float, the root-mean square error
    '''
    return np.sqrt(np.sum((actual-predicted)**2)/len(actual))

# Build a reusable data pipeline
def select_columns(data, *columns):
    '''Select only columns passed as arguments.'''
    return data.loc[:, columns]

def process_data_fm(data):
    data = remove_outliers(data, 'Gr_Liv_Area', upper=5000)
    
    data = fix_fireplace_qu(data)
    data = ohe_fireplace_qu(data)
    data = add_in_rich_neighborhood(data, rich_neighborhoods)
    data = add_total_bathrooms(data)

    data = data.fillna(0)
    data = select_columns(data, 
                          'SalePrice',
                          'Gr_Liv_Area',
                          'Year_Built',
                          'Garage_Area',
                          'TotalBathrooms',
                          'in_rich_neighborhood',
                          'Fireplace_Qu=Excellent',
                          'Fireplace_Qu=Fair',
                          'Fireplace_Qu=Good',
                          'Fireplace_Qu=No Fireplace',
                          'Fireplace_Qu=Poor'
                         )
    
    X = data.drop(['SalePrice'], axis = 1)
    y = data.loc[:, 'SalePrice']
    return X, y

# Data Visualizations
plt.rcParams['figure.figsize'] = (12, 9)
plt.rcParams['font.size'] = 12

# Plot the relationship between Sales Price and Total # of Bathrooms
training_data_no_outliers = remove_outliers(training_data, 'Gr_Liv_Area', upper=5000)
training_data_with_bathrooms = add_total_bathrooms(training_data_no_outliers)
sns.boxplot(
    x='TotalBathrooms',
    y='SalePrice',
    data=training_data_with_bathrooms.sort_values('TotalBathrooms'))
plt.title('Relationship between the # of Total Bathrooms and Sales Price of Houses')
plt.show()

# Plot the relationship between Sales Price and Above-Ground Living Area in sq. ft
sns.jointplot(
    x='Gr_Liv_Area', 
    y='SalePrice', 
    data=training_data,
    kind="reg",
    ratio=4,
    space=0,
    scatter_kws={
        's': 3,
        'alpha': 0.25
    },
    line_kws={
        'color': 'black'
    }
)
plt.show()

# Plot the relationship between Sales Price and Neighborhood
training_data_cleaned = pd.read_csv("data/ames_train_cleaned.csv")
fig, axs = plt.subplots(nrows=2, figsize=(12, 8))

sns.boxplot(
    x='Neighborhood',
    y='SalePrice',
    data=training_data_cleaned.sort_values('Neighborhood'),
    ax=axs[0]
)

sns.countplot(
    x='Neighborhood',
    data=training_data_cleaned.sort_values('Neighborhood'),
    ax=axs[1]
)

# Draw median price
axs[0].axhline(
    y=training_data_cleaned['SalePrice'].median(), 
    color='red',
    linestyle='dotted'
)

# Label the bars with counts
for patch in axs[1].patches:
    x = patch.get_bbox().get_points()[:, 0]
    y = patch.get_bbox().get_points()[1, 1]
    axs[1].annotate(f'{int(y)}', (x.mean(), y), ha='center', va='bottom')
    
# Format x-axes
axs[1].set_xticklabels(axs[1].xaxis.get_majorticklabels(), rotation=90)
axs[0].xaxis.set_visible(False)

# Narrow the gap between the plots
plt.subplots_adjust(hspace=0.01)
plt.show()

# Build and train our linear regression model
rich_neighborhoods = find_rich_neighborhoods(training_data_cleaned, 3, np.median)

training_data_fm = pd.read_csv('data/ames_train_cleaned.csv')
test_data_fm = pd.read_csv('data/ames_test_cleaned.csv')

X_train_fm, y_train_fm = process_data_fm(training_data_fm)
X_test_fm, y_test_fm = process_data_fm(test_data_fm)

final_model = lm.LinearRegression(fit_intercept=True)
final_model.fit(X_train_fm, y_train_fm)

y_predicted_train_fm = final_model.predict(X_train_fm)
y_predicted_test_fm = final_model.predict(X_test_fm)

training_rmse_fm = rmse(y_predicted_train_fm, y_train_fm)
test_rmse_fm = rmse(y_predicted_test_fm, y_test_fm)

print('Training RMSE: ', training_rmse_fm)
print('Test RMSE: ', test_rmse_fm)