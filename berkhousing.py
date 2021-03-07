import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn import linear_model as lm


plt.rcParams['figure.figsize'] = (12, 9)
plt.rcParams['font.size'] = 12

training_data = pd.read_csv("ames_train.csv")
test_data = pd.read_csv("ames_test.csv")

# 2000 observations and 82 features in training data
assert training_data.shape == (2000, 82)
# 930 observations and 81 features in test data
assert test_data.shape == (930, 81)
# SalePrice is hidden in the test data
assert 'SalePrice' not in test_data.columns.values
# Every other column in the test data should be in the training data
assert len(np.intersect1d(test_data.columns.values, 
                          training_data.columns.values)) == 81


fig, axs = plt.subplots(nrows=2)

sns.distplot(
    training_data['SalePrice'].values, 
    ax=axs[0]
)
sns.stripplot(
    training_data['SalePrice'].values, 
    jitter=0.4, 
    size=3,
    ax=axs[1],
    alpha=0.3
)
sns.boxplot(
    training_data['SalePrice'].values,
    width=0.3, 
    ax=axs[1],
    showfliers=False,
)

# Align axes
spacer = np.max(training_data['SalePrice']) * 0.05
xmin = np.min(training_data['SalePrice']) - spacer
xmax = np.max(training_data['SalePrice']) + spacer
axs[0].set_xlim((xmin, xmax))
axs[1].set_xlim((xmin, xmax))

# Remove some axis text
axs[0].xaxis.set_visible(False)
axs[0].yaxis.set_visible(False)
axs[1].yaxis.set_visible(False)

# Put the two plots together
plt.subplots_adjust(hspace=0)

# Adjust boxplot fill to be white
axs[1].artists[0].set_facecolor('white')

training_data['SalePrice'].describe()

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

training_data.loc[training_data['Gr_Liv_Area'] > 5000, 'PID'].values

def remove_outliers(data, variable, lower=-np.inf, upper=np.inf):
    """
    Input:
      data (data frame): the table to be filtered
      variable (string): the column with numerical outliers
      lower (numeric): observations with values lower than or equal to this will be removed
      upper (numeric): observations with values higher than or equal to this will be removed
    
    Output:
      a winsorized data frame with outliers removed
      
    Note: This function should not change mutate the contents of data.
    """  
    return data.loc[(data[variable] > lower) & (data[variable] < upper), :]

training_data_no_outliers = remove_outliers(training_data, 'Gr_Liv_Area', upper=5000)

def add_total_bathrooms(data):
    """
    Input:
      data (data frame): a data frame containing at least 4 numeric columns 
            Bsmt_Full_Bath, Full_Bath, Bsmt_Half_Bath, and Half_Bath
    """
    with_bathrooms = data.copy()
    bath_vars = ['Bsmt_Full_Bath', 'Full_Bath', 'Bsmt_Half_Bath', 'Half_Bath']
    weights = pd.Series([1, 1, 0.5, 0.5], index=bath_vars)
    with_bathrooms = with_bathrooms.fillna(0)
    with_bathrooms['TotalBathrooms'] = with_bathrooms['Bsmt_Full_Bath'] + with_bathrooms['Full_Bath'] + 1/2*(with_bathrooms['Bsmt_Half_Bath'] + with_bathrooms['Half_Bath'])
    return with_bathrooms

training_data_with_bathrooms = add_total_bathrooms(training_data_no_outliers)
training_data_with_bathrooms

sns.boxplot(
    x='TotalBathrooms',
    y='SalePrice',
    data=training_data_with_bathrooms.sort_values('TotalBathrooms'))
plt.title('Relationship between the # of Total Bathrooms and Sales Price of Houses')

# Load a fresh copy of the data and get its length
full_data = pd.read_csv("ames_train.csv")
full_data_len = len(full_data)
full_data.head()

train, val = train_test_split(full_data, test_size=0.2, random_state=42)

# Build a reusable data pipeline
def select_columns(data, *columns):
    """Select only columns passed as arguments."""
    return data.loc[:, columns]

def process_data_gm(data):
    """Process the data for a guided model."""
    data = remove_outliers(data, 'Gr_Liv_Area', upper=5000)
    
    # Transform Data, Select Features
    data = add_total_bathrooms(data)
    data = select_columns(data, 
                          'SalePrice', 
                          'Gr_Liv_Area', 
                          'Garage_Area',
                          'TotalBathrooms',
                         )
    
    # Return predictors and response variables separately
    X = data.drop(['SalePrice'], axis = 1)
    y = data.loc[:, 'SalePrice']
    
    return X, y

X_train, y_train = process_data_gm(train)
X_val, y_val = process_data_gm(val)

linear_model = lm.LinearRegression(fit_intercept=True)

y_fitted = linear_model.predict(X_train)
y_predicted = linear_model.predict(X_val)

def rmse(actual, predicted):
    """
    Calculates RMSE from actual and predicted values
    Input:
      actual (1D array): vector of actual values
      predicted (1D array): vector of predicted/fitted values
    Output:
      a float, the root-mean square error
    """
    return np.sqrt(np.sum((actual-predicted)**2)/len(actual))

training_error = rmse(y_train, y_fitted)
val_error = rmse(y_val, y_predicted)

X_train_no_bath = select_columns(val, 'Gr_Liv_Area', 'Garage_Area')

linear_model.fit(X_train_no_bath, y_val)
y_pred_no_bath = linear_model.predict(X_train_no_bath)

val_error_no_bath = rmse(y_val, y_pred_no_bath)
val_error_difference = val_error_no_bath - val_error
val_error_difference

training_data_cleaned = pd.read_csv("ames_train_cleaned.csv")

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

def find_rich_neighborhoods(data, n=3, metric=np.median):
    """
    Input:
      data (data frame): should contain at least a string-valued Neighborhood
        and a numeric SalePrice column
      n (int): the number of top values desired
      metric (function): function used for aggregating the data in each neighborhood.
        for example, np.median for median prices
    
    Output:
      a list of the top n richest neighborhoods as measured by the metric function
    """
    neighborhoods = data.groupby('Neighborhood').agg({'SalePrice':metric}).sort_values(by='SalePrice', ascending=False).head(n).index.tolist()
    return neighborhoods

rich_neighborhoods = find_rich_neighborhoods(training_data_cleaned, 3, np.median)
rich_neighborhoods

def add_in_rich_neighborhood(data, neighborhoods):
    """
    Input:
      data (data frame): a data frame containing a 'Neighborhood' column with values
        found in the codebook
      neighborhoods (list of strings): strings should be the names of neighborhoods
        pre-identified as rich
    Output:
      data frame identical to the input with the addition of a binary
      in_rich_neighborhood column
    """
    data_copy = data.copy()
    data_copy['in_rich_neighborhood'] = data_copy['Neighborhood'].isin(neighborhoods).astype(int)
    return data_copy

rich = find_rich_neighborhoods(training_data_cleaned, 3, np.median)
training_data_rich = add_in_rich_neighborhood(training_data_cleaned, rich)

missing_counts = training_data_rich.isnull().astype(int).sum().sort_values(ascending=False)
missing_counts

def fix_fireplace_qu(data):
    """
    Input:
      data (data frame): a data frame containing a Fireplace_Qu column.  Its values
                         should be limited to those found in the codebook
    Output:
      data frame identical to the input except with a refactored Fireplace_Qu column
    """
    new_data = data.copy()
    new_data['Fireplace_Qu'] = data['Fireplace_Qu'].replace(to_replace=['NA', 'TA', 'Ex', 'Gd', 'Fa', 'Po'], value=['No Fireplace', 'Average', 'Excellent', 'Good', 'Fair', 'Poor']).fillna('No Fireplace')
    return new_data
    
training_data_qu = fix_fireplace_qu(training_data_rich)

def ohe_fireplace_qu(data):
    """
    One-hot-encodes fireplace quality.  New columns are of the form Fireplace_Qu=QUALITY
    """
    vec_enc = DictVectorizer()
    vec_enc.fit(data[['Fireplace_Qu']].to_dict(orient='records'))
    fireplace_qu_data = vec_enc.transform(data[['Fireplace_Qu']].to_dict(orient='records')).toarray()
    fireplace_qu_cats = vec_enc.get_feature_names()
    fireplace_qu = pd.DataFrame(fireplace_qu_data, columns=fireplace_qu_cats)
    data = pd.concat([data, fireplace_qu], axis=1)
    data = data.drop(columns=fireplace_qu_cats[0])
    return data

training_data_ohe = ohe_fireplace_qu(training_data_qu)
training_data_ohe.filter(regex='Fireplace_Qu').head(10)

new_training_data = pd.read_csv("ames_train_cleaned.csv")

final_model = lm.LinearRegression(fit_intercept=True) # No need to change this!

# Build a resuable data pipeline for our new model
def process_data_fm(data):
    data = remove_outliers(data, 'Gr_Liv_Area', upper=5000)
    
    data = fix_fireplace_qu(data)
    data = ohe_fireplace_qu(data)
    data = add_in_rich_neighborhood(data, rich_neighborhoods)

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


training_data_fm = pd.read_csv('ames_train_cleaned.csv')
test_data_fm = pd.read_csv('ames_test_cleaned.csv')

X_train_fm, y_train_fm = process_data_fm(training_data_fm)
X_test_fm, y_test_fm = process_data_fm(test_data_fm)

final_model.fit(X_train_fm, y_train_fm)
y_predicted_train_fm = final_model.predict(X_train_fm)
y_predicted_test_fm = final_model.predict(X_test_fm)

training_rmse_fm = rmse(y_predicted_train_fm, y_train_fm)
test_rmse_fm = rmse(y_predicted_test_fm, y_test_fm)

training_rmse_fm