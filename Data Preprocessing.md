---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.4
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<!-- #region -->


# <b><div style='padding:25px;background-color:#9B2335;color:white;border-radius:4px;font-size:100%;text-align: center'>Industrial Copper Modeling<br></div>

<!-- #endregion -->

# Importing required libraries

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.stats import pearsonr
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import math
import scipy.stats as stats
from scipy.stats import boxcox
sns.set_theme(context='notebook', style='white', palette='dark', font='sans-serif', font_scale=1, color_codes=True, rc=None)  
import pickle
```

# Reading Dataset

```python
df=pd.read_excel('Copper_Set.xlsx')
```

# Understanding Data

```python
print(df.shape)
df.head(4)
```

1. Columns names are not unique. I prefer to use unique column names separated by underscore. I need  to change it.
2. Few of the variables data type is incorrect it seems. Need to check.
4. Material reference has a huge data point stars with multiple zero, must address this type of data points
5. ID column is not required. 


## updating column names

```python
df.columns
```

```python
col_name={'quantity tons':'quantity_tons','item type':'item_type','delivery date':'delivery_date'}
df.rename(columns=col_name, inplace=True)
df.columns
```

```python
df.info()
```

## Changing Data types

```python
print()
```

```python
df1 = df.copy()
```

```python
df1['quantity_tons'] = pd.to_numeric(df1['quantity_tons'], errors='coerce')
df1['status'] = df1['status'].astype('category')
df1['item_type'] = df1['item_type'].astype('category')
df1.item_date = pd.to_datetime(df1.item_date,format='%Y%m%d', errors='coerce')
df1.delivery_date = pd.to_datetime(df1.delivery_date,format='%Y%m%d', errors='coerce')
```

```python
print(f'{df1.quantity_tons.min()}  ---  {df1.quantity_tons.max()}')
print(f'{df1.thickness.min()}  ---  {df1.thickness.max()}')
print(f'{df1.width.min()}  ---  {df1.width.max()}')
print(f'{df1.selling_price.min()}  ---  {df1.selling_price.max()}')
```

## Data Types in the  Dataset

Understanding the data types in our dataset is crucial for effective analysis and visualization. Here's a breakdown of the common data types:

**1. Nominal Data:**

  * Represents categories or labels with no inherent order or ranking.
  * Operations: Not applicable for mathematical operations.
  *  Example: `customer`, `country`, `item type`, `status`
  
**2. Ordinal Data:**

  * Represents categories with a specific order or hierarchy.
  * Operations: Limited mathematical operations might be possible (e.g., finding the median rating).
  *  Example: `application`

**3. Discrete Data:**

  * Represents whole numbers that can only take on specific values within a range.
  * Operations: Mathematical operations like addition, subtraction, multiplication, and division are applicable on the whole numbers.
  * Example: `id`

**4. Continuous Data:**

  * Represents measurable quantities that can theoretically take on any value within a specific range.
  * Operations: Mathematical operations like addition, subtraction, multiplication, and division are applicable on the continuous values.
   * Examples: `quantity tons`, `thickness`, `width`, `selling_price`

**Remember:**

  * The data type interpretation can sometimes depend on the specific context and how you plan to use the data.
  * Dates can be considered nominal (specific dates) or ordinal (chronological order) depending on usage.
  * Statuses can be nominal (distinct categories) or ordinal (ordered progression) depending on the defined order.


```python
data_types = {
    'Nominal': [ 'customer','country',  'item_type', 'application', 'material_ref', 'product_ref'],
    'Ordinal': ['item_date','status','delivery_data'],
    'Continuous': ['quantity_tons', 'thickness', 'width', 'selling_price'],
}

```

```python
df1=df1.drop(columns =['id'])
pd.set_option('display.max_columns', 50)
pd.options.display.float_format = '{:.4f}'.format
df1.head()
```

```python
print(df1.info(),'\n')
print(df1.shape)
print('\n',df1.dtypes)
```

1. Incorrect Data types
2. Few variables  have  null values


## Checking statictical information of data

```python
df1.describe().T
```

1. Count confirms that the data has null values.
2. Quantity tons and Selling price has negative values which is not obvious.
3. Quantity tons and Selling price has more variance when comparing Min, percentiles, and max.


## Checking unique values of  variables

```python
print(df1.nunique(),'\n\n')
for col in df1.columns:
    print(f"{col} - {len(df1[col].unique())}") 
```

1. Country, Status, Item type, product reference,  and Application are having minimum class.
2. Again the differences confirms that the data has null values.


## Defining variables type so we can use them

```python
continuous=['quantity_tons', 'thickness', 'width', 'selling_price']
categorical =    ['application', 'customer', 'status', 'item_type', 'product_ref','country']

```

# Data Cleaning


## Cleaning Material reference variable as we seen  some of the data points are started with multiple zeros

```python
df1.loc[df1['material_ref'].str.contains('0{10,}', na=False), 'material_ref'] = np.nan
df1.loc[df1['customer']<30000000,'customer']=np.nan
```

## Removing negative values

```python
num=df1.select_dtypes(include=['number']).columns
for col in num:
    print(f"{col} -  {(df1[col] <= 0).sum()}")
```

```python
col_to_mask=['quantity_tons', 'selling_price']
df1[col_to_mask] =df1[col_to_mask] .mask(df1[col_to_mask] <= 1, np.nan)
```

```python
num=df1.select_dtypes(include=['number']).columns
for col in num:
    print(f"{col} -  {(df1[col] <= 0).sum()}")
```

## Checking data frame again after minor cleaning 

```python
df1.info()
```

1. We can see the data types are updated for further processing
2. Memory usage reduced from 18 MB to 15 MB


##  Checking Null values

```python
df1.isnull().sum()
```

```python
df1=df1.drop(columns =['material_ref'])
```

```python
df1.columns
```

## Checking the  Structure  data

```python
df1.country.value_counts()
```

```python
df1.item_type.value_counts()
```

```python
df1.status.value_counts()
```

```python
df1.application.value_counts()
```

```python
df1.product_ref.value_counts()
```

##  Imputing null datapoints

```python
print(f'{df1.quantity_tons.min()}  ---  {df1.quantity_tons.max()}')
print(f'{df1.thickness.min()}  ---  {df1.thickness.max()}')
print(f'{df1.width.min()}  ---  {df1.width.max()}')
print(f'{df1.selling_price.min()}  ---  {df1.selling_price.max()}')
```

```python
df1['status'] = df1.groupby('item_type',observed=False)['status'].transform(lambda x: x.fillna(x.mode().iloc[0]))
df1['customer'] = df1.groupby('item_type',observed=False)['customer'].transform(lambda x: x.fillna(x.mode().iloc[0]))
df1['country'] = df1.groupby('item_type',observed=False)['country'].transform(lambda x: x.fillna(x.mode().iloc[0]))
df1['application'] = df1.groupby('item_type',observed=False)['application'].transform(lambda x: x.fillna(x.mode().iloc[0]))
df1['thickness'] = df1.groupby('item_type',observed=False)['thickness'].transform(lambda x: x.fillna(x.median()))
df1['item_date'] = df1.groupby('item_type',observed=False)['item_date'].ffill()
df1['delivery_date'] = df1.groupby('item_type',observed=False)['delivery_date'].ffill()
df1['selling_price'] = df1.groupby('item_type',observed=False)['selling_price'].transform(lambda x: x.fillna(x.median()))
df1['quantity_tons'] = df1.groupby('item_type',observed=False)['quantity_tons'].transform(lambda x: x.fillna(x.median()))
```

```python
print(f'{df1.quantity_tons.min()}  ---  {df1.quantity_tons.max()}')
print(f'{df1.thickness.min()}  ---  {df1.thickness.max()}')
print(f'{df1.width.min()}  ---  {df1.width.max()}')
print(f'{df1.selling_price.min()}  ---  {df1.selling_price.max()}')
```

```python
print(df1.isnull().sum())
print(df1.shape)
```

## Updating datatypes for efficient usage

```python
print(f'{df1.quantity_tons.min()}  ---  {df1.quantity_tons.max()}')
print(f'{df1.thickness.min()}  ---  {df1.thickness.max()}')
print(f'{df1.width.min()}  ---  {df1.width.max()}')
print(f'{df1.selling_price.min()}  ---  {df1.selling_price.max()}')
```

```python
type_dict = {
    'item_date': 'datetime64[s]',
    'quantity_tons': 'float64',
    'customer': 'int32',
    'country': 'int16',
    'status': 'category',
    'item_type': 'category',
    'application': 'int64',
    'thickness': 'float32',
    'width': 'int32',
    'product_ref': 'int64',
    'selling_price': 'float64',
    'delivery_date': 'datetime64[s]'
}

df1 = df1.astype(type_dict)

df1.item_date = pd.to_datetime(df1.item_date,format='%Y%m%d', errors='coerce')
df1.delivery_date = pd.to_datetime(df1.delivery_date,format='%Y%m%d', errors='coerce')
```

```python
print(f'{df1.quantity_tons.min()}  ---  {df1.quantity_tons.max()}')
print(f'{df1.thickness.min()}  ---  {df1.thickness.max()}')
print(f'{df1.width.min()}  ---  {df1.width.max()}')
print(f'{df1.selling_price.min()}  ---  {df1.selling_price.max()}')
```

## Checking dataframe after null treatment & data type adjustment

```python
print(df1.isnull().sum().sum(),'\n\n',df1.shape,'\n\n',df1.head(3),'\n\n')
df1.info()
```

# Exploratory Data Analysis.  Before handling Outliers and Skewness

```python
def univar_num(df, col):
    plt.figure(figsize=(16, 8))
    df = df.dropna(subset=[col])


    mean=df[col].mean()
    median=df[col].median()
    mode=df[col].mode()[0]

    plt.subplot(2, 3, 1)
    sns.histplot(data=df, x=col, kde=True, bins=50,color='maroon',fill=True)
    plt.title(f'Histogram for {col}')
    plt.axvline(mean, color='maroon', label='Mean')
    plt.axvline(median, color='black', label='Median')
    plt.axvline(mode, color='darkgreen', label='Mode')
    
    plt.legend()
    plt.tight_layout()
    
    plt.subplot(2, 3, 2)
    sns.boxplot(data=df, x=col, color='lightgrey',legend=True)
    plt.xticks(rotation=45)
    plt.title(f'Box Plot for {col}')
    plt.axvline(mean, color='maroon', label='Mean')
    plt.axvline(median, color='black', label='Median')
    plt.axvline(mode, color='darkgreen', label='Mode')
    plt.legend()
    plt.tight_layout()

    
    plt.subplot(2, 3, 3)
    stats.probplot(df[col], dist="norm", plot=plt)
    plt.gca().get_lines()[1].set_color('maroon')
    plt.gca().get_lines()[0].set_color('darkgreen')
    plt.title(f'QQ Plot for {col}')

    plt.tight_layout()
    
    plt.tight_layout()
    plt.show()
```

## Continuous Variables Distribution

```python
color_good = '\033[32m'  # Dark Green for good skew
color_okay = '\033[92m'  # Light Green for okay skew
color_bad = '\033[91m'  # Maroon for bad skew
color_neutral = '\033[0m'  # Reset color

skewed_col_before_fix = []

for i in continuous:
    univar_num(df1,i) 

    skew_val = df1[i].skew()
    
    if -0.5 <= skew_val <= 0.5:
        color = color_good  # Dark Green for near-zero skew
    elif 0 < skew_val <= 0.5 or -0.5 < skew_val < 0:
        color = color_okay  # Light Green for slightly positive or slightly negative skew
    else:  # skew_val > 0.5 or skew_val < -0.5
        color = color_bad  # Maroon for significant skew
        skewed_col_before_fix.append(i)

    print(f"{color}Skew for {i} is {skew_val:.2f}{color_neutral}")

print(f"Skewed columns - {skewed_col_before_fix}")

```

## Comparing continuous Feature and continuousTarget - Regression problem

```python
continuous.remove('selling_price')
sns.set_theme(context='notebook', style='white', palette='dark', font='sans-serif', font_scale=2 ,color_codes=True, rc=None)  

total_var=len(continuous)
plot_col=3
plot_row=int(total_var/3)

fig, axes = plt.subplots(plot_row, plot_col, figsize=(50, 10))
axes = axes.flatten()
for i, var in  enumerate(continuous):
    sns.scatterplot(data=df1, x=var, y="selling_price", color='maroon',ax=axes[i],s=100)
    axes[i].set_title(f'Scatter plot: {var} vs Selling_Price')

for i in range(i + 1, len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()

```

## Categorical Variables Distribution

```python
for i in categorical:
    plt.figure(figsize=(20, 8))
    sns.countplot(data=df1, x=df1[i], hue=i, order = df1[i].value_counts().index.tolist()[::-1],palette='dark',legend=False)
    plt.xticks(rotation=25)
    plt.show()
```

## Comparing continuousFeature and categorical Target - Classification problem

```python

total_var=len(continuous)
plot_col=3
plot_row=math.ceil(total_var/3)

fig, axes = plt.subplots(plot_row, plot_col, figsize=(40, 8))
axes = axes.flatten()
for i, var in  enumerate(continuous):
    sns.scatterplot(data=df1, x=var, y="selling_price", hue='status',hue_order=['Won','Lost'],palette=['darkgreen','maroon'],ax=axes[i],s=100)
    axes[i].set_title(f'Scatter plot: {var} vs Selling_Price')

for i in range(i + 1, len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()
```

## Comparing categorical Feature and continuousTarget - Regression problem

```python
sns.set_theme( font_scale=1)  

for i in categorical:
    plt.figure(figsize=(20, 8))
    df1.groupby(i,observed=False)['selling_price'].mean().reset_index().sort_values('selling_price')
    sns.barplot(data=df1, x=df1[i], y='selling_price',hue=i, order = df1.groupby(i)['selling_price'].mean().reset_index().sort_values('selling_price')[i],legend=False,palette='dark',)
    plt.xticks(rotation=25)
    plt.title(f'Average Selling Price by {i}')
    plt.show()
```

## Comparing continuousFeature and categorical Target - Classification problem

```python
sns.color_palette('rocket',as_cmap=True)
for i in categorical:
    plt.figure(figsize=(20, 6))
    df1.groupby(i,observed=False)['selling_price'].mean().reset_index().sort_values('selling_price')
    sns.countplot(data=df1, x=df1[i],hue='status', hue_order=['Won','Lost'],legend=True,palette=['darkgreen','maroon'])
    plt.xticks(rotation=25)
    plt.title(f'Count plot for {i} in terms of status')
    plt.show()

```

## Date series Analysis

```python
col=['item_date', 'delivery_date']
total_var=len(col)
plot_col=2
plot_row=1

fig, axes = plt.subplots(plot_row, plot_col, figsize=(40, 10))
axes = axes.flatten()
for i, var in  enumerate(col):
    sns.lineplot(data=df1.groupby(var,observed=False)['selling_price'].mean().reset_index().sort_values('selling_price'),x=var,y='selling_price',ax=axes[i],color='maroon')
    axes[i].set_title(f'Line plot for {var} vs Selling_Price')

for i in range(i + 1, len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()
```

```python
cor_col = df1.select_dtypes(include='number')
correlation_matrix =cor_col.corr()
plt.figure(figsize=(9,6))
sns.heatmap(correlation_matrix,annot=True,fmt='.2f',cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
```

## Outlier Imputation

```python
print(f'{df1.quantity_tons.min()}  ---  {df1.quantity_tons.max()}')
print(f'{df1.thickness.min()}  ---  {df1.thickness.max()}')
print(f'{df1.width.min()}  ---  {df1.width.max()}')
print(f'{df1.selling_price.min()}  ---  {df1.selling_price.max()}')
```

```python
def outlier(df, column,iqr_fact):
    iqr = df[column].quantile(0.75) - df[column].quantile(0.25)
    upper_threshold = df[column].quantile(0.75) + (iqr_fact*iqr)
    lower_threshold = df[column].quantile(0.25) - (iqr_fact*iqr)
    df[column] = df[column].clip(lower_threshold, upper_threshold)
    return df

for i in ['quantity_tons','thickness','width','selling_price']:
    outlier(df1,i,1.5)
    
print(df1.isnull().sum().sum())
print(f'{df1.quantity_tons.min()}  ---  {df1.quantity_tons.max()}')
print(f'{df1.thickness.min()}  ---  {df1.thickness.max()}')
print(f'{df1.width.min()}  ---  {df1.width.max()}')
print(f'{df1.selling_price.min()}  ---  {df1.selling_price.max()}')
```

## Checking Duplicate

```python
print(df1.duplicated().sum())
```

```python
df1.loc[df1.duplicated(keep=False)].sort_values('selling_price')
```

```python
df1=df1.drop_duplicates()
```

```python
print(df1.info())
df1.shape
```

# Sales  lead


Total Leads: This represents the entire pool of potential customers who have expressed some interest in your product or service.
WON Leads: These leads have successfully converted into paying customers.
LOST Leads: These leads did not convert for various reasons.

```python
# Assuming your DataFrame is called df1

# Calculate total leads
total_leads = df1.shape[0]
print(total_leads)

# Count WON leads
won_leads = df1.query("status=='Won'").shape[0]
print(won_leads)
# Calculate conversion rate
conversion_rate = won_leads / total_leads

print("Overall conversion rate:", conversion_rate)

```

```python
# Analyze distribution across WON and LOST leads

# Example: Feature - item_type
won_item_types = df1.query("status=='Won'")['item_type'].value_counts()
lost_item_types = df1.query("status=='Lost'")['item_type'].value_counts()

# Print or visualize the counts (bar chart)
print("WON item types:", won_item_types)
print("LOST item types:", lost_item_types)

```

<!-- #region -->
| Data Types | Tests                | Purpose                                            |
|---|---|---|
| Categorical | Chi-Square Test       | Association between categorical variables              |
| Numerical   | T-Test (Independent/   | Compare means of two groups (independent) or before  |
|             | Paired)               | and after conditions (paired)                        |
| Numerical   | ANOVA                 | Compare means of three or more groups                  |


T-Test (Independent or Paired): These tests are used to compare means between two groups (independent) or before and after conditions (paired) in numerical data, but not for categorical variables like "country".
ANOVA (Analysis of Variance): Similar to t-tests, but it compares means of three or more groups, again assuming numerical data.
<!-- #endregion -->

```python
from scipy.stats import chi2_contingency

# Contingency table for country and status
contingency_table = pd.crosstab(df1['item_type'], df1['customer'])
print(contingency_table)
# Perform chi-square test
chi2, pval, _, _ = chi2_contingency(contingency_table.values)

# Check for significance (p-value < 0.05)
if pval < 0.05:
  print("There is a statistically significant association between country and lead conversion (p-value:", pval, ")")
else:
  print("No statistically significant association found (p-value:", pval, ")")

```

```python
df1.describe().T

```

```python
df1.dtypes
```

# Feature Engineering


## Deriving new features

```python
os.makedirs('pkls', exist_ok=True)
```

```python
df1['delivery_time_taken'] = (df1['delivery_date'] - df1['item_date']).abs().dt.days
df1['volume'] = (df1['quantity_tons'] * df1['thickness'] * df1['width']).astype(float)
df1['unit_price'] = (df1['selling_price'] / (df1['quantity_tons'] * df1['thickness'])).astype(float)

df1['delivery_day'] = df1['delivery_date'].dt.day
df1['delivery_month'] = df1['delivery_date'].dt.month
df1['delivery_year'] = df1['delivery_date'].dt.year

df1['item_day'] = df1['item_date'].dt.day
df1['item_month'] = df1['item_date'].dt.month
df1['item_year'] = df1['item_date'].dt.year

print(df1.isnull().sum())
print(df1.shape)

```

```python
continuous=['quantity_tons', 'thickness', 'width', 'selling_price','delivery_time_taken','volume','unit_price']
categorical =    ['application', 'customer', 'status', 'item_type', 'product_ref','country','delivery_day','delivery_month','delivery_year','item_day','item_month','item_year']
```

## Checking Skewness


## You can understand the presence and direction of skewness by looking at the skew values in the dictionary.

#### General interpretations of a skew value:
1. Positive Skew / Right Skew -  If the skew value is positive (greater than 0), the distribution is skewed to the right. A positive skew value greater than 1 suggests a moderately strong right skew.
2. Negative Skew / Left Skew - If the skew value is negative (less than 0), the distribution is skewed to the left. A negative skew value less than -1 suggests a moderately strong left skew.
3. Values closer to zero indicate a distribution closer to symmetry. A longer tail extending towards higher values.

```python
skew_dict=dict(df1.select_dtypes(include='number').skew())
skew_dict
```

```python
#Defining Colors  to print  skewness based on the nature
color_good = '\033[32m'  # Dark Green for good skew
color_okay = '\033[92m'  # Light Green for okay skew
color_bad = '\033[91m'  # Maroon for bad skew
color_neutral = '\033[0m'  # Reset color
```

## Experimenting a few types of data transformation techniques to see which suits best

```python
def boxcox_transform(x):
    # Perform Box-Cox transformation
    transformed_data,lmbda = boxcox(x)  # boxcox returns a tuple, we extract the first element
    return transformed_data,lmbda
```

```python
method_functions =  {
    'log': np.log,
    'square': np.sqrt,
    'rec': lambda x: 1 / x,
    'sig': lambda x: 1 / (1 + np.exp(-x)),
    'pow': lambda x: np.power(x, 2),
    'exp':lambda x: x**(1/5),
    'boxcox': boxcox_transform,
        }

def skewness_checker(df, column, method):
    normalized_df=pd.DataFrame()

    for method_name, func in method_functions.items():
        for col in column:

            new_column = f'{col}_{method_name}'
            if method_name in ['boxcox']:
                transformed_values = func(df[col])[0]
            else:
                transformed_values = func(df[col])
            normalized_df[new_column] = transformed_values

    return normalized_df
columns=['quantity_tons','thickness','width','volume','unit_price']
normalized_df=skewness_checker(df1, columns, method_functions)
```

```python
normalized_df
```

## Plotting transformed data to see visually.

```python
transformed_skewness = {}
for i in normalized_df.columns:
    univar_num(normalized_df,i)
    skews=normalized_df[i].skew()
    transformed_skewness.update({i:skews})
    color = color_neutral
    if -0.5 <= skews <= 0.5:
        color = color_good  # Dark Green for near-zero skew
    elif 0 < skews <= 0.5 or -0.5 < skews < 0:
        color = color_okay  # Light Green for slightly positive or slightly negative skew
    else:  # skew_val > 0.5 or skew_val < -0.5
        color = color_bad  # Maroon for significant skew
      
    print(f"{color}Skew for {i} is {skews:.2f}{color_neutral}")    
transformed_skewness
```

# Data Transformation

From the experiment above,we could see that the  boxcox data transformation method forms closer symetry

```python
df2=df1.copy()
```

```python
lambda_dict = {}

df2['quantity_tons_boxcox'] , lambda_dict['quantity_tons'] = boxcox_transform(df2['quantity_tons'])
df2['thickness_boxcox'] , lambda_dict['thickness']= boxcox_transform(df2['thickness'])
df2['width_boxcox'] , lambda_dict['width'] = boxcox_transform(df2['width'])
df2['selling_price_boxcox'] , lambda_dict['selling_price'] = boxcox_transform(df2['selling_price'])
df2['volume_boxcox'] ,lambda_dict['volume'] = boxcox_transform(df2['volume'])
df2['unit_price_boxcox'], lambda_dict['unit_price'] = boxcox_transform(df2['unit_price'])

with open(r'pkls\boxcox_lambdas.pkl', 'wb') as f:
    pickle.dump(lambda_dict, f)


df2.head()
```

## Changing Data types for transformed variable

```python
print(df2.thickness_boxcox.min(), '  -  ' , df2.thickness_boxcox.max())
print(df2.quantity_tons_boxcox.min(), '  -  ' , df2.quantity_tons_boxcox.max())
print(df2.width_boxcox.min(), '  -  ' , df2.width_boxcox.max())
print(df2.volume_boxcox.min(), '  -  ' , df2.volume_boxcox.max())
print(df2.selling_price_boxcox.min(), '  -  ' , df2.selling_price_boxcox.max())
print(df2.unit_price_boxcox.min(), '  -  ' , df2.unit_price_boxcox.max())
```

```python
df2['thickness_boxcox'] = df2['thickness_boxcox'].astype('int8')
df2['quantity_tons_boxcox'] = df2['quantity_tons_boxcox'].astype('int8')
df2['width_boxcox'] = df2['width_boxcox'].astype('int8')
df2['volume_boxcox'] = df2['volume_boxcox'].astype('int8')
df2['selling_price_boxcox'] = df2['selling_price_boxcox'].astype('int32')
df2['unit_price_boxcox'] = df2['unit_price_boxcox'].astype('int8')
```

# EDA After handling Outliers and Skewness

```python
color_positive = '\033[92m'  # Green for positive skew
color_negative = '\033[91m'  # Red for negative skew
color_neutral = '\033[0m'  # Reset color for near-zero skew
color_yellow = '\033[93m'  # Yellow for near-zero skew

skewed_col_after_fix = []

for i in ['width_boxcox', 'quantity_tons_boxcox','thickness_boxcox','volume_boxcox','unit_price_boxcox','selling_price_boxcox']:
    univar_num(df2,i) 
    skew_val = df2[i].skew()
    
    if -0.5 <= skew_val <= 0.5:
        color = color_good  # Dark Green for near-zero skew
    elif 0 < skew_val <= 0.5 or -0.5 < skew_val < 0:
        color = color_okay  # Light Green for slightly positive or slightly negative skew
    else:  # skew_val > 0.5 or skew_val < -0.5
        color = color_bad  # Maroon for significant skew
        skewed_col_after_fix.append(i)

    print(f"{color}Skew for {i} is {skew_val:.2f}{color_neutral}")
```

## Categorical Variables Distribution

```python
for i in categorical:
    plt.figure(figsize=(20, 5))
    sns.countplot(data=df2, x=df2[i], hue=i, order = df2[i].value_counts().index.tolist()[::-1],palette='dark',legend=False)
    plt.xticks(rotation=25)
    plt.show()
```

## Comparing continuous Feature and continuousTarget - Regression problem

```python
import math
sns.set_theme(context='notebook', style='white', palette='dark', font='sans-serif', font_scale=2 ,color_codes=True, rc=None)  

total_var=len(continuous)
plot_col=3
plot_row=math.ceil(total_var / 3)

fig, axes = plt.subplots(plot_row, plot_col, figsize=(50, 35))
axes = axes.flatten()
for i, var in  enumerate(continuous):
    sns.scatterplot(data=df2, x=var, y="selling_price", color='maroon',ax=axes[i],s=100)
    axes[i].set_title(f'Scatter plot: {var} vs Selling_Price')

for i in range(i + 1, len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()

```

## Comparing continuousFeature and categorical Target - Classification problem

```python
total_var=len(continuous)
plot_col=3
plot_row=math.ceil(total_var/3)

fig, axes = plt.subplots(plot_row, plot_col, figsize=(40, 35))
axes = axes.flatten()
for i, var in  enumerate(continuous):
    sns.scatterplot(data=df2, x=var, y="selling_price", hue='status',hue_order=['Won','Lost'],palette=['darkgreen','maroon'],ax=axes[i],s=100)
    axes[i].set_title(f'Scatter plot: {var} vs Selling_Price')

for i in range(i + 1, len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()

```

## Comparing categorical Feature and continuousTarget - Regression problem

```python
sns.set_theme( font_scale=1)  
for i in categorical:
    plt.figure(figsize=(25, 5))
    data=df2.groupby(i,observed=False)['selling_price'].median().reset_index().sort_values('selling_price')
    sns.barplot(data=data, x=data[i], y=data['selling_price'],hue=i, order =data[i],legend=False,palette='dark',)
    plt.xticks(rotation=25)
    plt.title(f'Average Selling Price by {i}')
    plt.show()
```

## Comparing continuousFeature and categorical Target - Classification problem

```python

```

```python
sns.color_palette('rocket',as_cmap=True)
for i in categorical:
    if i !='status':
        plt.figure(figsize=(20, 5))
        df2.groupby(i,observed=False)['status'].count().reset_index().sort_values('status')
        sns.countplot(data=df2, x=df2[i],hue='status', hue_order=['Won','Lost'],legend=True,palette=['darkgreen','maroon'])
        plt.xticks(rotation=25)
        plt.title(f'Count plot for {i} in terms of status')
        plt.show()
```

```python
cor_col = df2[['width_boxcox', 'quantity_tons_boxcox','thickness_boxcox','unit_price_boxcox','volume_boxcox','selling_price','country','application','thickness','delivery_time_taken','delivery_month','item_month']]
correlation_matrix = cor_col.corr()
plt.figure(figsize=(10,4))
sns.heatmap(correlation_matrix,annot=True,fmt='.2f',cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
```

```python
df3=df2.copy()
```

```python
print(df3.status.unique())
print(df3.status.value_counts())
```

```python
print(df3.item_type.unique())
print(df3.item_type.value_counts())
```

## Statistical Significance of Correlations for numerical features:

```python
for column in df3.select_dtypes(include='number').columns:
    if column != 'selling_price' and column != 'delivery_day':
        corr, p_value = pearsonr(df3[column], df3['selling_price'])
        print(f'Correlation between transformed_status and {column}: {corr}, p-value: {p_value}')
```

```python
sns.set_theme(font_scale=2)
continuous.append('product_ref')

continuous_var=(len(continuous))
plot_col=3
plot_row=math.ceil(continuous_var/3)

fig, axes = plt.subplots(plot_row, plot_col, figsize=(40, 35))
axes = axes.flatten()

for i, var in enumerate(continuous):
    sns.boxenplot(data=df3,x='status',y=var,hue='status',ax=axes[i],legend=False,order=['Won','Lost'],hue_order=['Won','Lost'],palette=['darkgreen','maroon'])
    axes[i].set_title(f'Cat plot: {var} vs Status')
     
for i in range(i + 1, len(axes)):
    fig.delaxes(axes[i])
    
plt.tight_layout()
plt.show()
```

```python
for i in categorical:
    if i  !='customer':
        grouped = df3.groupby([i,'status'],observed=False).size().unstack(fill_value=0)
        grouped =grouped[['Won', 'Lost']]
        grouped.plot(kind='bar',stacked=True,legend=False,color=['darkgreen', 'maroon'],figsize=(18, 5))
```

```python
df3.info()
```

```python
# df3.to_csv('ml_ready.csv',index=False)
df3=pd.read_csv('ml_ready.csv')
```

# Classification Method  Prediction  (Won/Lose) model building


## Dropping uneccesary Datapoints & Columns

```python
wanted_statuses = ['Won', 'Lost']  # Adjust as needed
df4 = df3[df3['status'].isin(wanted_statuses)]
print(df4.isnull().sum().sum(),'\n\n',df4.shape)
```

```python
DD=pd.crosstab(df4['status'],df4['item_type'],margins=False)
```

```python
DD
```

```python
from sklearn.metrics import mutual_info_score

mi = mutual_info_score(df4['status'], df4['item_type'])
print(f'Mutual Information: {mi}')

```

```python
from scipy.stats import chi2_contingency
stat, p, dof, expected = chi2_contingency(DD)

print(f'Stats - {stat} \n\n p_value - {p} \n\n dof  -  {dof} \n\n expected - {expected}')
```

```python
import matplotlib.pyplot as plt
from statsmodels.graphics.mosaicplot import mosaic

# Create the figure (unpacking not needed)
fig = plt.figure(figsize=(20, 8))  # Set desired figure size

# Create the mosaic plot using the figure object
ax = mosaic(df4, ['item_type', 'status'], ax=fig.add_subplot(111),label_rotation=45)

# Add a title to the plot
plt.title('Mosaic Plot of item_typr  by status')

# Display the plot
plt.show()

```

## Chi-Square Test of Independence for categorical Features:

```python
def cramers_v(confusion_matrix):
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

for feature in df4.select_dtypes(include='category').columns:
    confusion_matrix = pd.crosstab(df4[feature], df4['status'])
    chi2, p, dof, ex = chi2_contingency(confusion_matrix)
    cramers_v_value = cramers_v(confusion_matrix)
    
    print(f"Feature: {feature}")
    print(f"Chi-Square test p-value: {p}")
    print(f"Cramér's V: {cramers_v_value}")
    print("-" * 30)
```

```python
df5=df4.copy()
```

```python
df5.loc[:, 'transformed_status'] = df5['status'].map({'Won':1,'Lost':0,})

ohe= OneHotEncoder(sparse_output=False)
ohe.fit(df5[['item_type']])
item_type_encoded = ohe.transform(df5[['item_type']])
new_column_names = ohe.get_feature_names_out(['item_type'])
encode_item=pd.DataFrame(item_type_encoded, columns=new_column_names,dtype=np.int8,index=None)

df5 = df5.reset_index(drop=True)
encode_item = encode_item.reset_index(drop=True)
df6 = pd.concat([df5, encode_item], axis=1)
```

```python
with open(r'pkls\\Class_ohe.pkl', 'wb') as f:
    pickle.dump(ohe,f)
```

```python
print(df5.isnull().sum().sum(),'\n\n',df5.shape)
print(encode_item.isnull().sum().sum(),'\n\n',encode_item.shape)
print(df6.isnull().sum().sum(),'\n\n',df6.shape)
```

```python

```

```python
df6.head(3)
```

```python
sns.set_theme(font_scale=1)
cor_col = df6.select_dtypes(include='number')
correlation_matrix = cor_col.corr()
plt.figure(figsize=(30,15))
sns.heatmap(correlation_matrix,annot=True,fmt='.2f',cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
correlation_matrix
```

###  Cheking Statistical Significance of Correlations after Encoding:

```python
for column in df6.select_dtypes(include='number').columns:
    if column != 'transformed_status' and column != 'delivery_day':
        corr, p_value = pearsonr(df6[column], df6['transformed_status'])
        print(f'Correlation between transformed_status and {column}: {corr}, p-value: {p_value}')
```

```python
target_corr = df6.select_dtypes(include='number').corr().abs()['transformed_status'].sort_values(ascending=False)
target_corr
```

```python
df6.columns
```

```python
col_to_drop=['selling_price','delivery_day','thickness','unit_price','width','quantity_tons','item_date','delivery_date','status','item_type','volume',]

df6=df6.drop(columns=col_to_drop)
```

### Checking Data Frame

```python
df6.head()
```

```python
df6.tail()
```

```python
print(df6.shape,'\n\n', df6.info())
```

```python
df6.describe().T
```

```python
sns.set_theme(font_scale=1)
cor_col = df6.select_dtypes(include='number')
correlation_matrix = cor_col.corr()
plt.figure(figsize=(30,10))
sns.heatmap(correlation_matrix,annot=True,fmt='.2f',cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
correlation_matrix
```

## Importing Necessary libraries

```python

from sklearn.preprocessing import StandardScaler, OneHotEncoder , RobustScaler
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV,KFold, cross_val_score
from sklearn.metrics import accuracy_score,auc,roc_curve,confusion_matrix,classification_report,precision_score,recall_score,f1_score,roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from imblearn.combine import SMOTETomek
import pickle
from colorama import Fore, Style

color_positive = Fore.GREEN
reset_color = Style.RESET_ALL
```

## CheckingTarget variable unique values
The size of  the Positive (yes) data point is almost 4 times Negative (no)  datapoint size. Looks like we have imbalance data. 

```python
print(df6.transformed_status.unique())
df6.transformed_status.value_counts()
```

## Preparing Target and Features and balancing data points

```python
df6.to_csv('before_class.csv', index=False)
df6=pd.read_csv('before_class.csv')
```

```python
df6
```

### Pair Plot

```python
sns.pairplot(data=df6.sample(10000),hue='transformed_status')
```



```python
df6.columns
```

## Splitting data for Train and Test and Addressing Class Imbalance.

1. Used SMOTETomek method from imbalance combine module to balance the data points to have same size of Positive and Negative data points.
2. By balancing the datapoints we can train model well in both aspects.

```python
x_train, x_test, y_train, y_test = train_test_split(df6.drop('transformed_status', axis=1), df6[['transformed_status']].values.ravel(), test_size=0.2, random_state=42)

smote_tomek = SMOTETomek(random_state=42)
x_train, y_train = smote_tomek.fit_resample(x_train, y_train) 
```

```python
print(x_train.shape,'----', y_train.shape)
print(x_test.shape,'----', y_test.shape)
```

```python
y_series = pd.Series(y_train)
y_value_counts = y_series.value_counts()
y_value_counts
```

 Now our data is balanced

```python
y_series = pd.Series(y_test)
y_value_counts = y_series.value_counts()
y_value_counts
```

## Scaling data

```python
df6.iloc[1]
```

```python

robust_scaler = StandardScaler().fit(x_train)
x_train=robust_scaler.transform(x_train)
x_test=robust_scaler.transform(x_test)

with open(r'pkls\\scale_class.pkl', 'wb') as f:
    pickle.dump(robust_scaler,f)
```

```python
x_train[0]
```

```python
df6.country.value_counts().index.tolist()
```

```python
df6.shape
```

## Model Building and Evaluation

Our  target class is Binary. We cannot use Logistic regression alogorithm since there is no Linear Relationship with Feature and Target. I am using it just to know how it is working with non-linear and complex data for my learning purpose. We can use Tree and distance based alogorithms.


## Logistic Regression

```python
params = [
    {
        'penalty': ['l2'], 
        'C': [0.01, 0.1, 1, 10, 100], 
        'solver': ['lbfgs', 'newton-cg', 'libliner'],
        'max_iter': [100, 200, 500]
    }
]

LR_grid = GridSearchCV(LogisticRegression(),param_grid=params,scoring='accuracy',cv=5,verbose=3, n_jobs=1)
LR_grid.fit(x_train,y_train)
```

```python
print(f"LR Best Score - {LR_grid.best_score_}\n\nLR Best Params - {LR_grid.best_params_}\n\nLR Best Estimater - {LR_grid.best_estimator_} \n\nLR Best Index - {LR_grid.best_index_} ")
```

```python
LR_grid= LogisticRegression(penalty='l2',C=1,solver='newton-cg',max_iter=100)
LR_grid.fit(x_train,y_train)
y_preds=LR_grid.predict(x_test)
y_preds_train = LR_grid.predict(x_train)

print(f"{color_positive}Confusion Matrix Test {reset_color} -- {confusion_matrix(y_test,y_preds)}")
print(f'{color_positive}Confusion Matrix Train{reset_color} - {confusion_matrix(y_train,y_preds_train)}\n')

print(f'{color_positive}Accuracy Test {reset_color} - {accuracy_score(y_test,y_preds)}')
print(f'{color_positive}Accuracy Train{reset_color} - {accuracy_score(y_train,y_preds_train)}\n')

print(f'{color_positive}Precision Test {reset_color} - {precision_score(y_test,y_preds)}')
print(f'{color_positive}Precision Train{reset_color} - {precision_score(y_train,y_preds_train)}\n')

print(f'{color_positive}Recall Test- {reset_color} {recall_score(y_test,y_preds)}')
print(f'{color_positive}Recall Train{reset_color} - {recall_score(y_train,y_preds_train)}\n')

print(f'{color_positive}F1_score Test- {reset_color} {f1_score(y_test,y_preds)}')
print(f'{color_positive}F1_score Train{reset_color} - {f1_score(y_train,y_preds_train)}\n')

print('-'*40)

fpr, tpr, thresholds = roc_curve(y_test, y_preds)
auc = roc_auc_score(y_test, y_preds)
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve for Logistic Regression' )
plt.legend(loc="lower right")
plt.show()
```

## KNN Classifier

```python
params = {
    'n_neighbors': [2,3,4],
    'weights': ['uniform', 'distance'],
    'metric': ['manhattan', 'chebyshev', ]
}

knn_grid = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=params, cv=5, scoring='accuracy',verbose=3, n_jobs=1)
knn_grid.fit(x_train,y_train)
```

```python
print(f"KNN Best Score - {knn_grid.best_score_}\n\nKNN Best Params - {knn_grid.best_params_}\n\nKNN Best Estimater - {knn_grid.best_estimator_} \n\nKNN Best Index - {knn_grid.best_index_} ")
```

I used n_neighbors as 4 and  weights  as distabce as per the Gridsearch CV but it is overfitting and biased to trained data. So i did Cross Validation separatly for n_neighbors and weights alone and the results shows 3 and uniform  is the best. 

```python
knn_classifier =KNeighborsClassifier(n_neighbors=2)
knn_classifier.fit(x_train,y_train)
y_preds=knn_classifier.predict(x_test)
y_preds_train = knn_classifier.predict(x_train)

print(f"{color_positive}Confusion Matrix Test {reset_color} -- {confusion_matrix(y_test,y_preds)}")
print(f'{color_positive}Confusion Matrix Train{reset_color} - {confusion_matrix(y_train,y_preds_train)}\n')

print(f'{color_positive}Accuracy Test {reset_color} - {accuracy_score(y_test,y_preds)}')
print(f'{color_positive}Accuracy Train{reset_color} - {accuracy_score(y_train,y_preds_train)}\n')

print(f'{color_positive}Precision Test {reset_color} - {precision_score(y_test,y_preds)}')
print(f'{color_positive}Precision Train{reset_color} - {precision_score(y_train,y_preds_train)}\n')

print(f'{color_positive}Recall Test- {reset_color} {recall_score(y_test,y_preds)}')
print(f'{color_positive}Recall Train{reset_color} - {recall_score(y_train,y_preds_train)}\n')

print(f'{color_positive}F1_score Test- {reset_color} {f1_score(y_test,y_preds)}')
print(f'{color_positive}F1_score Train{reset_color} - {f1_score(y_train,y_preds_train)}\n')

fpr, tpr, thresholds = roc_curve(y_test, y_preds)
auc = roc_auc_score(y_test, y_preds)
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve for KNN' )
plt.legend(loc="lower right")
plt.show()
print('-'*40)
```

## Decision Tree


![image.png](attachment:image.png)
![image-2.png](attachment:image-2.png)

```python
params = {
    'max_depth': [ 3,5,7,11],
    'min_samples_split': [2, 4,6],
    #'max_features': [ 'sqrt', 'log2'],
    'min_samples_leaf': [1,5,10],
    #'criterion': ['entropy', 'gini'],
    #'splitter': ['best', 'random']
}

DT_grid = GridSearchCV(estimator= DecisionTreeClassifier(), param_grid=params, scoring='accuracy', cv=5, verbose=3, n_jobs=1)
DT_grid.fit(x_train,y_train)
```

```python
print(f"DT Best Score - {DT_grid.best_score_}\n\n DT Best Params - {DT_grid.best_params_}\n\nDT Best Estimater - {DT_grid.best_estimator_} \n\nDT Best Index - {DT_grid.best_index_} ")
```

```python
from sklearn.tree import plot_tree

DT_Classifier =DecisionTreeClassifier(max_depth=11,min_samples_leaf=1,min_samples_split=4,random_state=42,)
#criterion='entropy', splitter='best',max_features='sqrt'
DT_Classifier.fit(x_train,y_train)
y_preds=DT_Classifier.predict(x_test)
y_preds_train = DT_Classifier.predict(x_train)

print(f"{color_positive}Confusion Matrix Test {reset_color} -- {confusion_matrix(y_test,y_preds)}")
print(f'{color_positive}Confusion Matrix Train{reset_color} - {confusion_matrix(y_train,y_preds_train)}\n')

print(f'{color_positive}Accuracy Test {reset_color} - {accuracy_score(y_test,y_preds)}')
print(f'{color_positive}Accuracy Train{reset_color} - {accuracy_score(y_train,y_preds_train)}\n')

print(f'{color_positive}Precision Test {reset_color} - {precision_score(y_test,y_preds)}')
print(f'{color_positive}Precision Train{reset_color} - {precision_score(y_train,y_preds_train)}\n')

print(f'{color_positive}Recall Test- {reset_color} {recall_score(y_test,y_preds)}')
print(f'{color_positive}Recall Train{reset_color} - {recall_score(y_train,y_preds_train)}\n')

print(f'{color_positive}F1_score Test- {reset_color} {f1_score(y_test,y_preds)}')
print(f'{color_positive}F1_score Train{reset_color} - {f1_score(y_train,y_preds_train)}\n')

fpr, tpr, thresholds = roc_curve(y_test, y_preds)
auc = roc_auc_score(y_test, y_preds)
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve for Logistic Regression' )
plt.legend(loc="lower right")
plt.show()
print('-'*40)
```

```python
# Plot the decision tree
plt.figure(figsize=(20,10))  # Adjust the figure size as needed
plot_tree(DT_Classifier, filled=True, feature_names=df6.columns, class_names=['Class 0', 'Class 1'])  
plt.show()
```

## Extra Tree Classifier

```python
params = {
    
    'max_depth':[4,5],
    'min_samples_split': [2,3],    
        'max_features': ['sqrt','log2'],
        'min_samples_leaf': [1,2],
        'criterion':['gini','entropy','log_loss']
   }
```

```python
ET_grid = GridSearchCV(estimator= ExtraTreesClassifier(random_state=42), param_grid=params, scoring='accuracy', cv=5, verbose=3, n_jobs=1)
ET_grid.fit(x_train,y_train)
```

```python
print(f"ET Best Score - {ET_grid.best_score_}\n\n ET Best Params - {ET_grid.best_params_}\n\nET Best Estimater - {ET_grid.best_estimator_} \n\nET Best Index - {ET_grid.best_index_} ")
```

```python
ET_Classifier =ExtraTreesClassifier(max_depth=20,min_samples_leaf=1,min_samples_split=2,random_state=42,)
ET_Classifier.fit(x_train,y_train)
y_preds=ET_Classifier.predict(x_test)
y_preds_train = ET_Classifier.predict(x_train)

print(f"{color_positive}Confusion Matrix Test {reset_color} -- {confusion_matrix(y_test,y_preds)}")
print(f'{color_positive}Confusion Matrix Train{reset_color} - {confusion_matrix(y_train,y_preds_train)}\n')

print(f'{color_positive}Accuracy Test {reset_color} - {accuracy_score(y_test,y_preds)}')
print(f'{color_positive}Accuracy Train{reset_color} - {accuracy_score(y_train,y_preds_train)}\n')

print(f'{color_positive}Precision Test {reset_color} - {precision_score(y_test,y_preds)}')
print(f'{color_positive}Precision Train{reset_color} - {precision_score(y_train,y_preds_train)}\n')

print(f'{color_positive}Recall Test- {reset_color} {recall_score(y_test,y_preds)}')
print(f'{color_positive}Recall Train{reset_color} - {recall_score(y_train,y_preds_train)}\n')

print(f'{color_positive}F1_score Test- {reset_color} {f1_score(y_test,y_preds)}')
print(f'{color_positive}F1_score Train{reset_color} - {f1_score(y_train,y_preds_train)}\n')

fpr, tpr, thresholds = roc_curve(y_test, y_preds)
auc = roc_auc_score(y_test, y_preds)
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve for Logistic Regression' )
plt.legend(loc="lower right")
plt.show()
print('-'*40)
```

## Random Forest Classifier

```python
params = {
    
    'max_depth':[4,5,None],
    'min_samples_split': [2,3],    
        'min_samples_leaf': [1,2],

   }

RF_grid=GridSearchCV(estimator= RandomForestClassifier(random_state=42), param_grid=params, scoring='accuracy', cv=5, verbose=3, n_jobs=1)
RF_grid.fit(x_train,y_train)
```

```python
print(f"RF Best Score - {RF_grid.best_score_}\n\n RF Best Params - {RF_grid.best_params_}\n\nRF Best Estimater - {RF_grid.best_estimator_} \n\nRF Best Index - {RF_grid.best_index_} ")
```

```python
RF_Classifier =RandomForestClassifier(max_depth=15,min_samples_leaf=1,min_samples_split=2,)
RF_Classifier.fit(x_train,y_train)
y_preds=RF_Classifier.predict(x_test)
y_preds_train = RF_Classifier.predict(x_train)

print(f"{color_positive}Confusion Matrix Test {reset_color} -- {confusion_matrix(y_test,y_preds)}")
print(f'{color_positive}Confusion Matrix Train{reset_color} - {confusion_matrix(y_train,y_preds_train)}\n')

print(f'{color_positive}Accuracy Test {reset_color} - {accuracy_score(y_test,y_preds)}')
print(f'{color_positive}Accuracy Train{reset_color} - {accuracy_score(y_train,y_preds_train)}\n')

print(f'{color_positive}Precision Test {reset_color} - {precision_score(y_test,y_preds)}')
print(f'{color_positive}Precision Train{reset_color} - {precision_score(y_train,y_preds_train)}\n')

print(f'{color_positive}Recall Test- {reset_color} {recall_score(y_test,y_preds)}')
print(f'{color_positive}Recall Train{reset_color} - {recall_score(y_train,y_preds_train)}\n')

print(f'{color_positive}F1_score Test- {reset_color} {f1_score(y_test,y_preds)}')
print(f'{color_positive}F1_score Train{reset_color} - {f1_score(y_train,y_preds_train)}\n')

fpr, tpr, thresholds = roc_curve(y_test, y_preds)
auc = roc_auc_score(y_test, y_preds)
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve for Random Forest' )
plt.legend(loc="lower right")
plt.show()
print('-'*40)
```

## XGB Booster

```python
params = {
   # 'gamma': [0, 0.3, 0.5],
    'learning_rate': [0.05, 0.1,0.2,0.3],
    'max_depth': [ 5, 7,10],
    'min_child_weight': [3, 5, 1],
    'n_estimators': [500, 1000],
    #'reg_alpha': [0, 0.1, 0.5],
    #'reg_lambda': [1, 1.5, 2]
}


xgb_grid=GridSearchCV(estimator= XGBClassifier(random_state=42), param_grid=params, scoring='accuracy', cv=5, verbose=3, n_jobs=1)
xgb_grid.fit(x_train,y_train)
```

```python
print(f"XGB Best Score - {xgb_grid.best_score_}\n\n XGB Best Params - {xgb_grid.best_params_}\n\nXGB Best Estimater - {xgb_grid.best_estimator_} \n\nXGB Best Index - {xgb_grid.best_index_} ")
```

```python
xgb_Classifier =XGBClassifier(learning_rate=.43,max_depth=8,)
xgb_Classifier.fit(x_train,y_train)
y_preds=xgb_Classifier.predict(x_test)
y_preds_train = xgb_Classifier.predict(x_train)

print(f"{color_positive}Confusion Matrix Test {reset_color} -- {confusion_matrix(y_test,y_preds)}")
print(f'{color_positive}Confusion Matrix Train{reset_color} - {confusion_matrix(y_train,y_preds_train)}\n')

print(f'{color_positive}Accuracy Test {reset_color} - {accuracy_score(y_test,y_preds)}')
print(f'{color_positive}Accuracy Train{reset_color} - {accuracy_score(y_train,y_preds_train)}\n')

print(f'{color_positive}Precision Test {reset_color} - {precision_score(y_test,y_preds)}')
print(f'{color_positive}Precision Train{reset_color} - {precision_score(y_train,y_preds_train)}\n')

print(f'{color_positive}Recall Test- {reset_color} {recall_score(y_test,y_preds)}')
print(f'{color_positive}Recall Train{reset_color} - {recall_score(y_train,y_preds_train)}\n')

print(f'{color_positive}F1_score Test- {reset_color} {f1_score(y_test,y_preds)}')
print(f'{color_positive}F1_score Train{reset_color} - {f1_score(y_train,y_preds_train)}\n')

fpr, tpr, thresholds = roc_curve(y_test, y_preds)
auc = roc_auc_score(y_test, y_preds)
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve for Random Forest' )
plt.legend(loc="lower right")
plt.show()
print('-'*40)
```

## Classification Model Selection

From the above models, Random Forest and XG Booster proides good accuracy score and F1 score. So i am further evaluating model performance using Cross Validation


### Cross Validation for XG Booster Classifier.

```python
from sklearn.model_selection import KFold, cross_val_score
kf=KFold(n_splits=10,shuffle=True,random_state=42)
xgb_cv_score = cross_val_score(xgb_Classifier,x_train,y_train, cv=kf)
print(f'Cross validations scores \n\n {xgb_cv_score}\n\n')
print(f'Cross validations scores mean \n\n {np.mean(xgb_cv_score)}')
```

### Cross Validation for Random Forest Classifier.

```python
kf=KFold(n_splits=10,shuffle=True,random_state=42)
RF_cv_score = cross_val_score(RF_Classifier,x_train,y_train, cv=kf)
print(f'Cross validations scores \n\n {RF_cv_score}')
print(f'Cross validations scores mean \n\n {np.mean(RF_cv_score)}')
```

## Finalizing Classification Model


From the Cross Validation report, we can sence that the XGB Classifier provides stable accuracy score. So i choose XGB for this classification problem.

```python
xgb_Classifier =XGBClassifier(learning_rate=1.25,n_estimators=365,min_child_weight=1.5,max_depth=7)
# max_depth=10,learning_rate=0.239,min_child_weight=1,n_estimators=365
xgb_Classifier.fit(x_train,y_train)
y_preds=xgb_Classifier.predict(x_test)
y_preds_train = xgb_Classifier.predict(x_train)

print(f"{color_positive}Confusion Matrix Test {reset_color} -- {confusion_matrix(y_test,y_preds)}")
print(f'{color_positive}Confusion Matrix Train{reset_color} - {confusion_matrix(y_train,y_preds_train)}\n')

print(f'{color_positive}Accuracy Test {reset_color} - {accuracy_score(y_test,y_preds)}')
print(f'{color_positive}Accuracy Train{reset_color} - {accuracy_score(y_train,y_preds_train)}\n')

print(f'{color_positive}Precision Test {reset_color} - {precision_score(y_test,y_preds)}')
print(f'{color_positive}Precision Train{reset_color} - {precision_score(y_train,y_preds_train)}\n')

print(f'{color_positive}Recall Test- {reset_color} {recall_score(y_test,y_preds)}')
print(f'{color_positive}Recall Train{reset_color} - {recall_score(y_train,y_preds_train)}\n')

print(f'{color_positive}F1_score Test- {reset_color} {f1_score(y_test,y_preds)}')
print(f'{color_positive}F1_score Train{reset_color} - {f1_score(y_train,y_preds_train)}\n')

fpr, tpr, thresholds = roc_curve(y_test, y_preds)
auc = roc_auc_score(y_test, y_preds)
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve for Random Forest' )
plt.legend(loc="lower right")
plt.show()
print('-'*40)
```

```python

```

```python

```

```python

```

### Pickling XGB Classifier

```python
with open(r'pkls\\xgb_classifier_model.pkl', 'wb') as f:
    pickle.dump(xgb_Classifier,f)
```

## Checking Feature Importances

```python
from xgboost import plot_importance

# feature_names = x_train.columns
# importances = xgb_Classifier.feature_importances_


# feature_importances_df = pd.DataFrame({
#     'Feature': feature_names,
#     'Importance': importances
# })
# feature_importances_df = feature_importances_df.sort_values(by='Importance', ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(20, 8)) 

ax1 = axes[0]
ax2 = axes[1]

plot_importance(xgb_Classifier, max_num_features=10,ax=ax1)
plt.title('XGBoost Feature Importance')


sns.barplot(x='Importance', y='Feature', data=feature_importances_df,ax=ax2)
plt.title('Feature Importances')
plt.tight_layout()

plt.show()
```

## Checking Feature Importances using Recursive Feature Elimination (RFE) 

```python
from sklearn.feature_selection import RFE
selector=RFE(xgb_Classifier,n_features_to_select=8, step=1)
selector=selector.fit(x,y)
```

```python
selector.support_
```

```python
selector.ranking_
```

```python
features = selector.ranking_ <= 4
selected_x_train = x_train[:, features]  
selected_x_test = x_test[:, features]  
```

```python

xgb_Classifier =XGBClassifier(max_depth=10,learning_rate=0.3,min_child_weight=1,n_estimators=365)
xgb_Classifier.fit(selected_x_train,y_train)
y_preds=xgb_Classifier.predict(selected_x_test)
y_preds_train = xgb_Classifier.predict(selected_x_train)

print(f"{color_positive}Confusion Matrix Test {reset_color} -- {confusion_matrix(y_test,y_preds)}")
print(f'{color_positive}Confusion Matrix Train{reset_color} - {confusion_matrix(y_train,y_preds_train)}\n')

print(f'{color_positive}Accuracy Test {reset_color} - {accuracy_score(y_test,y_preds)}')
print(f'{color_positive}Accuracy Train{reset_color} - {accuracy_score(y_train,y_preds_train)}\n')

print(f'{color_positive}Precision Test {reset_color} - {precision_score(y_test,y_preds)}')
print(f'{color_positive}Precision Train{reset_color} - {precision_score(y_train,y_preds_train)}\n')

print(f'{color_positive}Recall Test- {reset_color} {recall_score(y_test,y_preds)}')
print(f'{color_positive}Recall Train{reset_color} - {recall_score(y_train,y_preds_train)}\n')

print(f'{color_positive}F1_score Test- {reset_color} {f1_score(y_test,y_preds)}')
print(f'{color_positive}F1_score Train{reset_color} - {f1_score(y_train,y_preds_train)}\n')
```

<!-- DT_model - - DecisionTreeClassifier()

Confusion Matrix  -- [[27992   938]
 [  990  7693]]

Accuracy  - 0.9487411267380959

Precision  - 0.8913219789132197

Recall -  0.8859841068755039

F1_score -  0.8886450271456625
----------------------------------------

RF_model - - RandomForestClassifier()

Confusion Matrix  -- [[28326   604]
 [ 1142  7541]]

Accuracy  - 0.9535798792970516

Precision  - 0.9258440761203192

Recall -  0.8684786364159852

F1_score -  0.8962443546470169
----------------------------------------

ET_model - - ExtraTreesClassifier()

Confusion Matrix  -- [[28264   666]
 [ 1026  7657]]

Accuracy  - 0.9550155531332252

Precision  - 0.9199807761624414

Recall -  0.8818380743982495

F1_score -  0.9005057038692226
----------------------------------------

AB_model - - AdaBoostClassifier(algorithm='SAMME')

Confusion Matrix  -- [[27708  1222]
 [ 6702  1981]]

Accuracy  - 0.7893281578177758

Precision  - 0.6184826724945364

Recall -  0.2281469538178049

F1_score -  0.3333333333333333
----------------------------------------

GB_model - - GradientBoostingClassifier()

Confusion Matrix  -- [[27735  1195]
 [ 5121  3562]]

Accuracy  - 0.8320793342727248

Precision  - 0.7487912549926424

Recall -  0.41022688011056085

F1_score -  0.5300595238095238
----------------------------------------

LR_model - - LogisticRegression()

Confusion Matrix  -- [[27675  1255]
 [ 7115  1568]]

Accuracy  - 0.7774705553930822

Precision  - 0.5554374778604322

Recall -  0.18058274789819187

F1_score -  0.2725534503737181
----------------------------------------

XGB_model - - XGBClassifier(base_score=None, booster=None, callbacks=None,
              colsample_bylevel=None, colsample_bynode=None,
              colsample_bytree=None, device=None, early_stopping_rounds=None,
              enable_categorical=False, eval_metric=None, feature_types=None,
              gamma=None, grow_policy=None, importance_type=None,
              interaction_constraints=None, learning_rate=None, max_bin=None,
              max_cat_threshold=None, max_cat_to_onehot=None,
              max_delta_step=None, max_depth=None, max_leaves=None,
              min_child_weight=None, missing=nan, monotone_constraints=None,
              multi_strategy=None, n_estimators=None, n_jobs=None,
              num_parallel_tree=None, random_state=None, ...)

Confusion Matrix  -- [[27886  1044]
 [ 1647  7036]]

Accuracy  - 0.9284555871640124

Precision  - 0.8707920792079208

Recall -  0.8103190141656109

F1_score -  0.8394678756785778
---------------------------------------- -->


```python
    fig, axes = plt.subplots(1,2 , figsize=(20, 6))
for j,i in enumerate(['item_type','status']):
    df=df3_data[i].value_counts().reset_index(name='count')
    sizes=df['count'].to_list()
    labels=df[i].tolist()
    axes[j].pie(sizes, labels=sizes, autopct='%1.1f%%', startangle=140 ,labeldistance=.44,radius=1.2,explode=[.1]* len(sizes),shadow=True)
    axes[j].legend(labels,loc="center left", bbox_to_anchor=(1, 0.5))
    
    axes[j].set_title(f'Pie Chart: Distribution of {i}\n')

plt.tight_layout()
plt.show()  
```

# Regression Method  Prediction  (Selling Price) model building

```python
df3_data=pd.read_csv('ml_ready.csv')
```

```python
df3_data.quantity_tons.max()
```

```python
df3_data.columns
```

```python
# Assuming df3_data is your DataFrame containing the data
data_types = {
    'Nominal': ['customer', 'country', 'item_type', 'application', 'material_ref', 'product_ref','delivery_day','delivery_month','delivery_year','item_day','item_month','item_year'],
    'Ordinal': ['item_date', 'status', 'delivery_date'],
    'Continuous': ['quantity_tons', 'thickness', 'width', 'quantity_tons_boxcox','width_boxcox', 'thickness_boxcox', 'volume_boxcox', 'unit_price_boxcox', 'delivery_time_taken']
}

num_continuous_features = len(data_types['Continuous'])  # Get the number of continuous features
cols = 3  # Set the number of columns for the subplots
rows = math.ceil(num_continuous_features / cols)  # Calculate the required number of rows

fig, axes = plt.subplots(rows, cols, figsize=(20, 12))  # Create subplots based on calculated rows and columns
axes = axes.flatten()  # Flatten the axes array for easier indexing

for idx, var in enumerate(data_types['Continuous']):
    sns.scatterplot(data=df3_data, x=var, y='selling_price', color='maroon', ax=axes[idx])
    axes[idx].set_title(f'Scatter Plot: {var} vs selling_price \n')

# Hide any unused subplots
for i in range(idx + 1, len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()

```

```python
plt.figure(figsize=(14,6))
sns.kdeplot(df3_data[df3_data.status=='Won'].selling_price,color='green',label='Won',linewidth=2.5,fill=True)
sns.kdeplot(df3_data[df3_data.status=='Lost'].selling_price,color='maroon',label='Won',linewidth=2.5,fill=True)
sns.kdeplot(df3_data.selling_price,color='orange',label='Won',linewidth=2.5,fill=True)
plt.title('Status VS price distribution for price')
plt.xlim(100,1500)
plt.show()
```

## Importing necessary Libraries

```python
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.preprocessing import RobustScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.model_selection import KFold, cross_val_score
import pickle
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error, explained_variance_score,root_mean_squared_error
from sklearn.preprocessing import StandardScaler,LabelEncoder
import time
from colorama import Fore, Style
pd.set_option('display.max_columns', 50)
color_positive = Fore.GREEN
reset_color = Style.RESET_ALL
```

```python
df3_data.head(3)
```

## Encoding the data

```python
df4_data = df3_data.copy()
```

```python
app_encode=df3_data.groupby('application',observed=False)['selling_price'].median().rank(method='dense').reset_index().sort_values('selling_price')
app_encode_list = dict(zip(app_encode['application'], app_encode['selling_price'].astype(int)))
print(app_encode_list)
```

```python
product_encode=df3_data.groupby('product_ref',observed=False)['selling_price'].median().rank(method='dense').reset_index().sort_values('selling_price')
product_ref_encode_list = dict(zip(product_encode['product_ref'], product_encode['selling_price'].astype(int)))
print(product_ref_encode_list)
```

```python
bb=df3_data.groupby('country',observed=False)['selling_price'].median().rank().reset_index().sort_values('selling_price')
country_encode_list = dict(zip(bb['country'], bb['selling_price'].astype(int)))
print(country_encode_list)
```

```python

df4_data['application'] = df4_data['application'].map({58: 0, 68: 2, 28: 3, 56: 4, 25: 5, 59: 6, 69: 7, 15: 8, 39: 9, 4: 10, 3: 11, 5: 12, 22: 13, 40: 14, 10: 15, 26: 16, 66: 17, 65: 18, 70: 19, 67: 20, 27: 21, 20: 22, 19: 23, 29: 24, 79: 25, 2: 26, 42: 27, 41: 28, 38: 29, 99: 32})


df4_data['item_type'] = df4_data['item_type'].map({'WI':0,'PL':1,'Others':2,'IPL':3,'S':4,'W':5,'SLAWR':7})

# ohe= OneHotEncoder(sparse_output=False)
# ohe.fit(df4_data[['item_type']])
# item_type_encoded = ohe.transform(df4_data[['item_type']])
# new_column_names = ohe.get_feature_names_out(['item_type'])
# encode_item=pd.DataFrame(item_type_encoded, columns=new_column_names,dtype=np.int8,index=None)

ohe1= OneHotEncoder(sparse_output=False)
ohe1.fit(df4_data[['status']])
status_encoded = ohe1.transform(df4_data[['status']])
assert status_encoded.shape[1] == len(ohe1.get_feature_names_out(['status'])), "Mismatch in encoded status dimensions."
new_column_names = ohe1.get_feature_names_out(['status'])
encode_status=pd.DataFrame(status_encoded, columns=new_column_names,dtype=np.int8,index=None)

df4_data = df4_data.reset_index(drop=True)
# encode_item = encode_item.reset_index(drop=True)
encode_status = encode_status.reset_index(drop=True)
df4_data = pd.concat([df4_data,encode_status], axis=1)

# df4_data['product_ref'] = df4_data['product_ref'].map({640665: 1, 1693867563: 2, 1671876026: 3, 628377: 4, 1670798778: 5, 640405: 6, 1722207579: 7, 1671863738: 8, 640400: 9, \
#                                                        1665584662: 10, 1690738206: 11, 164141591: 12, 1721130331: 12, 628112: 13, 1693867550: 14, 929423819: 14, 164337175: 15, 628117: 16, 1332077137: 17, 1282007633: 18,\
#                                                        1665572032: 19, 1668701725: 20, 1668701376: 21, 164336407: 22, 1665572374: 23, 1668701718: 24, 1668701698: 25, 1665584320: 26, 1665584642: 27, 1690738219: 28, 611733: 29, 611993: 30, 611728: 31})

# df4_data['country'] = df4_data['country'].map({107: 1, 89: 2, 80: 3, 40: 4, 79: 5, 77: 6, 26: 7, 39: 8, 78: 9, 28: 10, 27: 11, 84: 12, 25: 13, 32: 14, 30: 15, 38: 16, 113: 19})


df4_data['application'] = df4_data['application'].astype('int8')
df4_data['item_type'] = df4_data['item_type'].astype('int8')
# df4_data['product_ref'] = df4_data['product_ref'].astype('int8')
# df4_data['country'] = df4_data['country'].astype('int8')
# df3_data['country'] = df3_data['country'].astype('int16')
# df3_data['transformed_app'] = df3_data['transformed_app'].astype('int32')
# df3_data['transformed_customer'] = df3_data['transformed_customer'].astype('int32')
# df3_data['transformed_product_ref'] = df3_data['transformed_product_ref'].astype('int32')

```

```python
with open(r'pkls\\Reg_ohe.pkl', 'wb') as f:
    pickle.dump(ohe1,f)
```

```python
df4_data.columns
```

```python
col_to_drop=['status','thickness', 'width', 'selling_price','volume','unit_price','item_date','delivery_date','quantity_tons',]
df4_data=df4_data.drop(columns =col_to_drop)

correlation_matrix = df4_data.select_dtypes(include='number').corr()
plt.figure(figsize=(30,10))
sns.heatmap(correlation_matrix,annot=True,fmt='.2f',cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
```


#### Multicollinity

3. Application - Widthbox
4. Del time - country, Delivery year, item month,  item year, Status lost, status won
4.  qty - Item_type, volumn
5. thick - width , volumn
6. width - app, thick,volumne
7. 


```python
target_corr = df4_data.select_dtypes(include='number').corr()['selling_price_boxcox'].abs().sort_values(ascending=False)[1:]
target_corr
```

```python
col_to_drop=['delivery_day','unit_price_boxcox']
df5_data=df4_data.drop(columns =col_to_drop).copy()
```

```python
correlation_matrix = df5_data.select_dtypes(include='number').corr()
plt.figure(figsize=(18,8))
sns.heatmap(correlation_matrix,annot=True,fmt='.2f',cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
```

```python
x= df5_data.drop(['selling_price_boxcox'], axis=1)
y=df5_data[['selling_price_boxcox']].values.ravel()
```

## Train, Test split and Scaling

```python
x.head(2)
```

```python
y[0:5]
```

```python
x_train,x_test,y_train,y_test  = train_test_split(x,y, test_size=0.2 ,  random_state=42)

scale_reg = StandardScaler().fit(x_train)
x_train=scale_reg.transform(x_train)
x_test=scale_reg.transform(x_test)

assert not np.any(np.isnan(x_train)), "NaN values found in scaled training data."
assert not np.any(np.isnan(x_test)), "NaN values found in scaled test data."

with open(r'pkls\\scale_reg.pkl', 'wb') as f:
    pickle.dump(scale_reg,f)

```

```python
print(x.shape, '\n\n',y.shape)
```

```python

```

## Lasso Regressor

```python
params = {
        'alpha': [1.0, 0.5,1.5,2], 
        'fit_intercept': [True, False],  
        'copy_X': [True, False],  
        'max_iter':[500,1000,1500],
        'warm_start': [True, False],  
        'selection':['cyclic', 'random']
    }

Lasso_grid = GridSearchCV(Lasso(),param_grid=params,scoring='r2',cv=5,verbose=3, n_jobs=1,error_score='raise',random_state=42)
Lasso_grid.fit(x_train,y_train)
```

```python
print(f"LR Best Score - {Lasso_grid.best_score_}\n\nLR Best Params - {Lasso_grid.best_params_}\n\nLR Best Estimater - {Lasso_grid.best_estimator_} \n\nLR Best Index - {Lasso_grid.best_index_} ")
```

```python
start_time = time.time()
Lasso_grid= Lasso(alpha=0.5,copy_X=True,fit_intercept=True,max_iter=100)
Lasso_grid.fit(x_train,y_train)
y_preds=Lasso_grid.predict(x_test)
y_preds_train = Lasso_grid.predict(x_train)

training_time = time.time() - start_time

print(f"{color_positive}Training Time: {training_time:.2f} seconds\n")

print(f"{color_positive}Test Mean Squared Error {reset_color} -- {mean_squared_error(y_test,y_preds):.2f}")
print(f"{color_positive}Train Mean Squared Error {reset_color} -- {mean_squared_error(y_train,y_preds_train):.2f}\n")

print(f"{color_positive}Test Root Mean Squared Error {reset_color} -- {root_mean_squared_error(y_test,y_preds):.2f}")
print(f"{color_positive}Train Root Mean Squared Error {reset_color} -- {root_mean_squared_error(y_train,y_preds_train):.2f}\n")

print(f"{color_positive}Test R2 Score {reset_color} -- {r2_score(y_test,y_preds):.2f}")
print(f"{color_positive}Train R2 Score {reset_color} -- {r2_score(y_train,y_preds_train):.2f}\n")    

print(f"{color_positive}Test Mean Absolute Error {reset_color} -- {mean_absolute_error(y_test,y_preds):.2f}")
print(f"{color_positive}Train Mean Absolute Error {reset_color} -- {mean_absolute_error(y_train,y_preds_train):.2f}\n")    

print(f"{color_positive}Test Median Absolute Error {reset_color} -- {median_absolute_error(y_test,y_preds):.2f}")
print(f"{color_positive}Train Median Absolute Error {reset_color} -- {median_absolute_error(y_train,y_preds_train):.2f}\n")    

print(f"{color_positive}Test Explained Var {reset_color} -- {explained_variance_score(y_test,y_preds):.2f}")
print(f"{color_positive}Train Explained Var {reset_color} -- {explained_variance_score(y_train,y_preds_train):.2f}\n")    
print('-'*50)
```

## KNN Regressor    

```python
params = {
    'n_neighbors': [2,3,4,5],
    'weights': ['uniform', 'distance'],
    'metric': ['manhattan', 'chebyshev', ],
    'algorithm':['ball_tree','kd_tree','brute','auto'],
    'leaf_size':[30,25,35]
}

knn_grid = GridSearchCV(estimator=KNeighborsRegressor(), param_grid=params, cv=5, scoring='r2',verbose=3, n_jobs=1,error_score='raise')
knn_grid.fit(x_train,y_train)
```

```python
print(f"KNN Best Score - {knn_grid.best_score_}\n\nKNN Best Params - {knn_grid.best_params_}\n\nKNN Best Estimater - {knn_grid.best_estimator_} \n\nKNN Best Index - {knn_grid.best_index_} ")
```

```python
start_time = time.time()
knn_regressor =KNeighborsRegressor(n_neighbors=5,metric='manhattan',weights='uniform')
knn_regressor.fit(x_train,y_train)
y_preds=knn_regressor.predict(x_test)
y_preds_train = knn_regressor.predict(x_train)

training_time = time.time() - start_time

print(f"{color_positive}Training Time: {training_time:.2f} seconds\n")

print(f"{color_positive}Test Mean Squared Error {reset_color} -- {mean_squared_error(y_test,y_preds):.2f}")
print(f"{color_positive}Train Mean Squared Error {reset_color} -- {mean_squared_error(y_train,y_preds_train):.2f}\n")

print(f"{color_positive}Test Root Mean Squared Error {reset_color} -- {root_mean_squared_error(y_test,y_preds):.2f}")
print(f"{color_positive}Train Root Mean Squared Error {reset_color} -- {root_mean_squared_error(y_train,y_preds_train):.2f}\n")

print(f"{color_positive}Test R2 Score {reset_color} -- {r2_score(y_test,y_preds):.2f}")
print(f"{color_positive}Train R2 Score {reset_color} -- {r2_score(y_train,y_preds_train):.2f}\n")    

print(f"{color_positive}Test Mean Absolute Error {reset_color} -- {mean_absolute_error(y_test,y_preds):.2f}")
print(f"{color_positive}Train Mean Absolute Error {reset_color} -- {mean_absolute_error(y_train,y_preds_train):.2f}\n")    

print(f"{color_positive}Test Median Absolute Error {reset_color} -- {median_absolute_error(y_test,y_preds):.2f}")
print(f"{color_positive}Train Median Absolute Error {reset_color} -- {median_absolute_error(y_train,y_preds_train):.2f}\n")    

print(f"{color_positive}Test Explained Var {reset_color} -- {explained_variance_score(y_test,y_preds):.2f}")
print(f"{color_positive}Train Explained Var {reset_color} -- {explained_variance_score(y_train,y_preds_train):.2f}\n")    
print('-'*50)
```

```python
new_sample = np.array([[30394817,78,5,10,1670798778,94,7,2021,29,3,5,0,14,16,2,0,0,0,1,0,0,0,0,0]])
# new_sample_ohe = ohe.transform(new_sample[:, [7]]).toarray()
# new_sample_be = ohe2.transform(new_sample[:, [8]]).toarray()
# new_sample = np.concatenate((new_sample[:, [0,1,2, 3, 4, 5, 6,]], new_sample_ohe, new_sample_be), axis=1)
new_sample1 = scale_class.transform(new_sample)
new_pred = knn_regressor.predict(new_sample1)
print('Predicted selling price:', (new_pred))
```

## Decision Tree Regressor

```python
param_grid = {
    
    'max_depth':[4,5,7,8],
    'min_samples_split': [2, 5,8],    
       'max_features': ['sqrt', 'log2'],
        'min_samples_leaf': [2, 4,6,15],
       'splitter':['best','random'],
       'criterion':['poisson','squared_error','friedman_mse',],
       'max_leaf_nodes':[None,6,3]
}


DT_grid = GridSearchCV(estimator=DecisionTreeRegressor(random_state=42), param_grid=param_grid, scoring='r2', cv=3, verbose=3, n_jobs=1,error_score='raise')

DT_grid.fit(x_train, y_train)

```

```python
print("Best Parameters:", DT_grid.best_params_)
print("Best Negative Mean Squared Error:", DT_grid.best_score_)
print(f"DT Best Score - {DT_grid.best_score_}\n\nDT Best Params - {DT_grid.best_params_}\n\nDT Best Estimater - {DT_grid.best_estimator_} \n\nDT Best Index - {DT_grid.best_index_} ")
```

```python
DT_Regressor =DecisionTreeRegressor(max_depth=10,criterion='poisson',min_samples_leaf=3,random_state=42)#plitter='best',max_leaf_nodes=None)
#criterion='poisson',max_depth=10,min_samples_leaf=6,min_samples_split=8,,
DT_Regressor.fit(x_train,y_train)
y_preds=DT_Regressor.predict(x_test)
y_preds_train = DT_Regressor.predict(x_train)

# print(f"{color_positive}Training Time: {training_time:.2f} seconds\n")

print(f"{color_positive}Test Mean Squared Error {reset_color} -- {mean_squared_error(y_test,y_preds):.2f}")
print(f"{color_positive}Train Mean Squared Error {reset_color} -- {mean_squared_error(y_train,y_preds_train):.2f}\n")

print(f"{color_positive}Test Root Mean Squared Error {reset_color} -- {root_mean_squared_error(y_test,y_preds):.2f}")
print(f"{color_positive}Train Root Mean Squared Error {reset_color} -- {root_mean_squared_error(y_train,y_preds_train):.2f}\n")

print(f"{color_positive}Test R2 Score {reset_color} -- {r2_score(y_test,y_preds):.2f}")
print(f"{color_positive}Train R2 Score {reset_color} -- {r2_score(y_train,y_preds_train):.2f}\n")    

print(f"{color_positive}Test Mean Absolute Error {reset_color} -- {mean_absolute_error(y_test,y_preds):.2f}")
print(f"{color_positive}Train Mean Absolute Error {reset_color} -- {mean_absolute_error(y_train,y_preds_train):.2f}\n")    

print(f"{color_positive}Test Median Absolute Error {reset_color} -- {median_absolute_error(y_test,y_preds):.2f}")
print(f"{color_positive}Train Median Absolute Error {reset_color} -- {median_absolute_error(y_train,y_preds_train):.2f}\n")    

print(f"{color_positive}Test Explained Var {reset_color} -- {explained_variance_score(y_test,y_preds):.2f}")
print(f"{color_positive}Train Explained Var {reset_color} -- {explained_variance_score(y_train,y_preds_train):.2f}\n")    
print('-'*50)
```

```python
new_sample = np.array([[30394817,78,5,10,1670798778,94,7,2021,29,3,5,0,14,16,2,0,0,0,1,0,0,0,0,0]])
# new_sample_ohe = ohe.transform(new_sample[:, [7]]).toarray()
# new_sample_be = ohe2.transform(new_sample[:, [8]]).toarray()
# new_sample = np.concatenate((new_sample[:, [0,1,2, 3, 4, 5, 6,]], new_sample_ohe, new_sample_be), axis=1)
new_sample1 = scale_class.transform(new_sample)
new_pred = DT_Regressor.predict(new_sample1)
print('Predicted selling price:', (new_pred))
```

## XG Boost Regressor

```python
param_grid = {
    'gamma': [0, 0.3, 0.5],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 7,12,10],
    'min_child_weight': [3, 5, 7],
    # 'n_estimators': [500, 1000],
    # 'reg_alpha': [0, 0.1, 0.5],
    # 'reg_lambda': [1, 1.5, 2]
}

xgb_grid = GridSearchCV(estimator=XGBRegressor(random_state=42), param_grid=param_grid, scoring='r2', cv=3, verbose=3, n_jobs=1,error_score='raise')

xgb_grid.fit(x_train, y_train)
```

```python
print(f"XGB Best Score - {xgb_grid.best_score_}\n\nXGB Best Params - {xgb_grid.best_params_}\n\nXGB Best Estimater - {xgb_grid.best_estimator_} \n\nXGB Best Index - {xgb_grid.best_index_} ")
```

```python
xgb_Reg =XGBRegressor(learning_rate=.561,min_child_weight=3)
# max_depth=11,learning_rate=0.06,colsample_bytree=0.8,subsample=1,n_estimators=239

xgb_Reg.fit(x_train,y_train)
y_preds=xgb_Reg.predict(x_test)
y_preds_train = xgb_Reg.predict(x_train)
# print(f"{color_positive}Training Time: {training_time:.2f} seconds\n")

print(f"{color_positive}Test Mean Squared Error {reset_color} -- {mean_squared_error(y_test,y_preds):.2f}")
print(f"{color_positive}Train Mean Squared Error {reset_color} -- {mean_squared_error(y_train,y_preds_train):.2f}\n")

print(f"{color_positive}Test Root Mean Squared Error {reset_color} -- {root_mean_squared_error(y_test,y_preds):.2f}")
print(f"{color_positive}Train Root Mean Squared Error {reset_color} -- {root_mean_squared_error(y_train,y_preds_train):.2f}\n")

print(f"{color_positive}Test R2 Score {reset_color} -- {r2_score(y_test,y_preds):.2f}")
print(f"{color_positive}Train R2 Score {reset_color} -- {r2_score(y_train,y_preds_train):.2f}\n")    

print(f"{color_positive}Test Mean Absolute Error {reset_color} -- {mean_absolute_error(y_test,y_preds):.2f}")
print(f"{color_positive}Train Mean Absolute Error {reset_color} -- {mean_absolute_error(y_train,y_preds_train):.2f}\n")    

print(f"{color_positive}Test Median Absolute Error {reset_color} -- {median_absolute_error(y_test,y_preds):.2f}")
print(f"{color_positive}Train Median Absolute Error {reset_color} -- {median_absolute_error(y_train,y_preds_train):.2f}\n")    

print(f"{color_positive}Test Explained Var {reset_color} -- {explained_variance_score(y_test,y_preds):.2f}")
print(f"{color_positive}Train Explained Var {reset_color} -- {explained_variance_score(y_train,y_preds_train):.2f}\n")    
print('-'*50)
```

```python
new_sample = np.array([[30394817,78,5,10,1670798778,94,7,2021,29,3,5,0,14,16,2,0,0,0,1,0,0,0,0,0]])
# new_sample_ohe = ohe.transform(new_sample[:, [7]]).toarray()
# new_sample_be = ohe2.transform(new_sample[:, [8]]).toarray()
# new_sample = np.concatenate((new_sample[:, [0,1,2, 3, 4, 5, 6,]], new_sample_ohe, new_sample_be), axis=1)
new_sample1 = scale_reg.transform(new_sample)
new_pred = xgb_Reg.predict(new_sample1)
Diff = 1125-new_pred
print('Predicted selling price:' ,(new_pred),'\n\n' ,Diff)
```

## Extra Tree Regressor

```python
params = {
    
    'max_depth':[4,5,8,10],
    'min_samples_split': [2,5],    
      #  'max_features': ['sqrt','log2'],
        'min_samples_leaf': [1,2,4],
        'criterion':['poisson','squared_error','friedman_mse',],
         'max_leaf_nodes':[None,6,3]
   }
ET_grid = GridSearchCV(estimator= ExtraTreesRegressor(random_state=42), param_grid=params, scoring='r2', cv=3, verbose=3, n_jobs=1,error_score='raise')
ET_grid.fit(x_train,y_train)
```

```python
print(f"ET Best Score - {ET_grid.best_score_}\n\nET Best Params - {ET_grid.best_params_}\n\nET Best Estimater - {ET_grid.best_estimator_} \n\nET Best Index - {ET_grid.best_index_} ")
```

```python
ET_Reg =ExtraTreesRegressor(max_depth=12,min_samples_leaf=1,min_samples_split=5,criterion='poisson',random_state=42,n_estimators=200)

ET_Reg.fit(x_train,y_train)
y_preds=ET_Reg.predict(x_test)
y_preds_train = ET_Reg.predict(x_train)
print(f"{color_positive}Training Time: {training_time:.2f} seconds\n")

print(f"{color_positive}Test Mean Squared Error {reset_color} -- {mean_squared_error(y_test,y_preds):.2f}")
print(f"{color_positive}Train Mean Squared Error {reset_color} -- {mean_squared_error(y_train,y_preds_train):.2f}\n")

print(f"{color_positive}Test Root Mean Squared Error {reset_color} -- {root_mean_squared_error(y_test,y_preds):.2f}")
print(f"{color_positive}Train Root Mean Squared Error {reset_color} -- {root_mean_squared_error(y_train,y_preds_train):.2f}\n")

print(f"{color_positive}Test R2 Score {reset_color} -- {r2_score(y_test,y_preds):.2f}")
print(f"{color_positive}Train R2 Score {reset_color} -- {r2_score(y_train,y_preds_train):.2f}\n")    

print(f"{color_positive}Test Mean Absolute Error {reset_color} -- {mean_absolute_error(y_test,y_preds):.2f}")
print(f"{color_positive}Train Mean Absolute Error {reset_color} -- {mean_absolute_error(y_train,y_preds_train):.2f}\n")    

print(f"{color_positive}Test Median Absolute Error {reset_color} -- {median_absolute_error(y_test,y_preds):.2f}")
print(f"{color_positive}Train Median Absolute Error {reset_color} -- {median_absolute_error(y_train,y_preds_train):.2f}\n")    

print(f"{color_positive}Test Explained Var {reset_color} -- {explained_variance_score(y_test,y_preds):.2f}")
print(f"{color_positive}Train Explained Var {reset_color} -- {explained_variance_score(y_train,y_preds_train):.2f}\n")    
print('-'*50)
```

```python
new_sample = np.array([[30394817,78,5,10,1670798778,94,7,2021,29,3,5,0,14,16,2,0,0,0,1,0,0,0,0,0]])
# new_sample_ohe = ohe.transform(new_sample[:, [7]]).toarray()
# new_sample_be = ohe2.transform(new_sample[:, [8]]).toarray()
# new_sample = np.concatenate((new_sample[:, [0,1,2, 3, 4, 5, 6,]], new_sample_ohe, new_sample_be), axis=1)
new_sample1 = scale_class.transform(new_sample)
new_pred = ET_Reg.predict(new_sample1)
Diff = 1125-new_pred
print('Predicted selling price:' ,(new_pred),'\n\n' ,Diff)
```

## Random Forest Regressor

```python
params = {
    
    'max_depth':[4,5,10],
    'min_samples_split': [2,3,4],    
        'min_samples_leaf': [1,2,4],

   }

RF_grid=GridSearchCV(estimator= RandomForestRegressor(random_state=42), param_grid=params, scoring='r2', cv=3, verbose=3, n_jobs=1)
RF_grid.fit(x_train,y_train)
```

```python
print(f"RF Best Score - {RF_grid.best_score_}\n\n RF Best Params - {RF_grid.best_params_}\n\nRF Best Estimater - {RF_grid.best_estimator_} \n\nRF Best Index - {RF_grid.best_index_} ")
```

```python
RF_Reg =RandomForestRegressor()
#max_depth=None,min_samples_leaf=1,min_samples_split=2,random_state=42,n_estimators=200

RF_Reg.fit(x_train,y_train)
y_preds=RF_Reg.predict(x_test)
y_preds_train = RF_Reg.predict(x_train)

print(f"{color_positive}Training Time: {training_time:.2f} seconds\n")

print(f"{color_positive}Test Mean Squared Error {reset_color} -- {mean_squared_error(y_test,y_preds):.2f}")
print(f"{color_positive}Train Mean Squared Error {reset_color} -- {mean_squared_error(y_train,y_preds_train):.2f}\n")

print(f"{color_positive}Test Root Mean Squared Error {reset_color} -- {root_mean_squared_error(y_test,y_preds):.2f}")
print(f"{color_positive}Train Root Mean Squared Error {reset_color} -- {root_mean_squared_error(y_train,y_preds_train):.2f}\n")

print(f"{color_positive}Test R2 Score {reset_color} -- {r2_score(y_test,y_preds):.2f}")
print(f"{color_positive}Train R2 Score {reset_color} -- {r2_score(y_train,y_preds_train):.2f}\n")    

print(f"{color_positive}Test Mean Absolute Error {reset_color} -- {mean_absolute_error(y_test,y_preds):.2f}")
print(f"{color_positive}Train Mean Absolute Error {reset_color} -- {mean_absolute_error(y_train,y_preds_train):.2f}\n")    

print(f"{color_positive}Test Median Absolute Error {reset_color} -- {median_absolute_error(y_test,y_preds):.2f}")
print(f"{color_positive}Train Median Absolute Error {reset_color} -- {median_absolute_error(y_train,y_preds_train):.2f}\n")    

print(f"{color_positive}Test Explained Var {reset_color} -- {explained_variance_score(y_test,y_preds):.2f}")
print(f"{color_positive}Train Explained Var {reset_color} -- {explained_variance_score(y_train,y_preds_train):.2f}\n")    
print('-'*50)
```

## Regression Model Selection

From the above models, XG Booster proides good R2  and other scores comparing to other models. So i am choosing  XG Boost Regressor. Let's check the model once again with Cross Validation using Kfold.

```python
xgb_Reg =XGBRegressor(learning_rate=.61)
kf=KFold(n_splits=10,shuffle=True,random_state=42)
xgb_cv_score = cross_val_score(xgb_Reg,x_train,y_train, cv=kf)
print(f'Cross validations scores \n\n {xgb_cv_score}\n\n')
print(f'Cross validations scores mean \n\n {np.mean(xgb_cv_score)}')
```

## Checking Feature Importances

```python
from xgboost import plot_importance

feature_names = x.columns
importances = xgb_Reg.feature_importances_


feature_importances_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
})
feature_importances_df = feature_importances_df.sort_values(by='Importance', ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(20, 8)) 

ax1 = axes[0]
ax2 = axes[1]

plot_importance(xgb_Reg,ax=ax1)
plt.title('XGBoost Feature Importance')


sns.barplot(x='Importance', y='Feature', data=feature_importances_df,ax=ax2)
plt.title('Feature Importances')
plt.tight_layout()

plt.show()
```

## Pickling Regression Model

```python
with open(r'pkls\\xgb_regression_model.pkl', 'wb') as f:
    pickle.dump(xgb_Reg,f)
```

<!-- #region -->


# <b><div style='padding:25px;background-color:#9B2335;color:white;border-radius:4px;font-size:100%;text-align: center'>END<br></div>

<!-- #endregion -->

```python

```

```python

```

```python
import time
importance={}
for name, model in reg_model:
    start_time = time.time() 
    print(f"{color_positive}{name} - - {model}\n")
    
    model.fit(x_train,y_train)
    y_preds=model.predict(x_test)
    y_preds_train=model.predict(x_train)
    training_time = time.time() - start_time
    if name not in ['KNN_model', 'Lasso_model']:
        importance[name] = model.feature_importances_
   
    
    print(f"{color_positive}Training Time: {training_time:.2f} seconds\n")

    print(f"{color_positive}Test Mean Squared Error {reset_color} -- {mean_squared_error(y_test,y_preds):.2f}")
    print(f"{color_positive}Train Mean Squared Error {reset_color} -- {mean_squared_error(y_train,y_preds_train):.2f}\n")

    print(f"{color_positive}Test Root Mean Squared Error {reset_color} -- {root_mean_squared_error(y_test,y_preds):.2f}")
    print(f"{color_positive}Train Root Mean Squared Error {reset_color} -- {root_mean_squared_error(y_train,y_preds_train):.2f}\n")
    
    print(f"{color_positive}Test R2 Score {reset_color} -- {r2_score(y_test,y_preds):.2f}")
    print(f"{color_positive}Train R2 Score {reset_color} -- {r2_score(y_train,y_preds_train):.2f}\n")    
    
    print(f"{color_positive}Test Mean Absolute Error {reset_color} -- {mean_absolute_error(y_test,y_preds):.2f}")
    print(f"{color_positive}Train Mean Absolute Error {reset_color} -- {mean_absolute_error(y_train,y_preds_train):.2f}\n")    
    
    print(f"{color_positive}Test Median Absolute Error {reset_color} -- {median_absolute_error(y_test,y_preds):.2f}")
    print(f"{color_positive}Train Median Absolute Error {reset_color} -- {median_absolute_error(y_train,y_preds_train):.2f}\n")    
    
    print(f"{color_positive}Test Explained Var {reset_color} -- {explained_variance_score(y_test,y_preds):.2f}")
    print(f"{color_positive}Train Explained Var {reset_color} -- {explained_variance_score(y_train,y_preds_train):.2f}\n")    
    print('-'*50)
```

```python

```

```python

```

```python
combined = np.column_stack((y_preds, y_test))
sd = pd.DataFrame(combined,columns=['y_preds','y_test'])
sd['diff'] = sd['y_test']  - sd['y_preds']
np.mean(sd['diff'])
```

```python
DT_Regressor =DecisionTreeRegressor(max_depth=12,min_samples_leaf=6,min_samples_split=8,random_state=42)
#max_depth': 8, 'max_features': 'sqrt', 'min_samples_leaf': 6, 'min_samples_split': 8, 'splitter': 'best'}
DT_Regressor.fit(x_train,y_train)
y_preds=DT_Regressor.predict(x_test)
y_preds_train = DT_Regressor.predict(x_train)

print(f"{color_positive}Training Time: {training_time:.2f} seconds\n")

print(f"{color_positive}Test Mean Squared Error {reset_color} -- {mean_squared_error(y_test,y_preds):.2f}")
print(f"{color_positive}Train Mean Squared Error {reset_color} -- {mean_squared_error(y_train,y_preds_train):.2f}\n")

print(f"{color_positive}Test Root Mean Squared Error {reset_color} -- {root_mean_squared_error(y_test,y_preds):.2f}")
print(f"{color_positive}Train Root Mean Squared Error {reset_color} -- {root_mean_squared_error(y_train,y_preds_train):.2f}\n")

print(f"{color_positive}Test R2 Score {reset_color} -- {r2_score(y_test,y_preds):.2f}")
print(f"{color_positive}Train R2 Score {reset_color} -- {r2_score(y_train,y_preds_train):.2f}\n")    

print(f"{color_positive}Test Mean Absolute Error {reset_color} -- {mean_absolute_error(y_test,y_preds):.2f}")
print(f"{color_positive}Train Mean Absolute Error {reset_color} -- {mean_absolute_error(y_train,y_preds_train):.2f}\n")    

print(f"{color_positive}Test Median Absolute Error {reset_color} -- {median_absolute_error(y_test,y_preds):.2f}")
print(f"{color_positive}Train Median Absolute Error {reset_color} -- {median_absolute_error(y_train,y_preds_train):.2f}\n")    

print(f"{color_positive}Test Explained Var {reset_color} -- {explained_variance_score(y_test,y_preds):.2f}")
print(f"{color_positive}Train Explained Var {reset_color} -- {explained_variance_score(y_train,y_preds_train):.2f}\n")    
print('-'*50)
```

```python
new_sample = np.array([[30394817,78,10,1670798778,94,7,2021,29,3,5,0,14,16,2,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0]])
# new_sample_ohe = ohe.transform(new_sample[:, [7]]).toarray()
# new_sample_be = ohe2.transform(new_sample[:, [8]]).toarray()
# new_sample = np.concatenate((new_sample[:, [0,1,2, 3, 4, 5, 6,]], new_sample_ohe, new_sample_be), axis=1)
new_sample1 = scale_class.transform(new_sample)
new_pred = DT_Regressor.predict(new_sample1)
print('Predicted selling price:', (new_pred))
```

Suggestions to Fix Overfitting:

Reduce the Complexity: Limit the depth of the tree using max_depth.
Pruning: Increase min_samples_split or min_samples_leaf

```python
fea_imp['Avg'] = fea_imp.mean(axis=1)
fea_imp.sort_values(by='XGB_model',ascending=False)
```

```python
x= df3_XGB.drop(['selling_price'], axis=1)
y=df3_XGB[['selling_price']].values.ravel()
```

```python
x_train,x_test,y_train,y_test  = train_test_split(x,y, test_size=0.2 ,  random_state=42)

scale_class = StandardScaler().fit(x_train)
x_train=scale_class.transform(x_train)
x_test=scale_class.transform(x_test)
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
```

```python

```

ou can try increasing the maximum depth to make the trees less complex and prevent them from capturing too much noise from the training data. 

```python

```

```python
print("Best Parameters:", grid_search.best_params_)
print("Best Negative Mean Squared Error:", grid_search.best_score_)
```

```python
import time
model = XGBRegressor(gamma= 0.5, learning_rate= 0.1, max_depth=12, n_estimators= 100,min_child_weight=7,reg_alpha=0.5,reg_lambda=1)
name = 'XGB Boost'
start_time = time.time() 
print(f"{color_positive}{name} - - {model}\n")


model.fit(x_train,y_train)
y_preds=model.predict(x_test)
y_preds_train=model.predict(x_train)
training_time = time.time() - start_time

print(f"{color_positive}Training Time: {training_time:.2f} seconds\n")

print(f"{color_positive}Test Mean Squared Error {reset_color} -- {mean_squared_error(y_test,y_preds):.2f}")
print(f"{color_positive}Train Mean Squared Error {reset_color} -- {mean_squared_error(y_train,y_preds_train):.2f}\n")

print(f"{color_positive}Test Root Mean Squared Error {reset_color} -- {root_mean_squared_error(y_test,y_preds):.2f}")
print(f"{color_positive}Train Root Mean Squared Error {reset_color} -- {root_mean_squared_error(y_train,y_preds_train):.2f}\n")

print(f"{color_positive}Test R2 Score {reset_color} -- {r2_score(y_test,y_preds):.2f}")
print(f"{color_positive}Train R2 Score {reset_color} -- {r2_score(y_train,y_preds_train):.2f}\n")    

print(f"{color_positive}Test Mean Absolute Error {reset_color} -- {mean_absolute_error(y_test,y_preds):.2f}")
print(f"{color_positive}Train Mean Absolute Error {reset_color} -- {mean_absolute_error(y_train,y_preds_train):.2f}\n")    

print(f"{color_positive}Test Median Absolute Error {reset_color} -- {median_absolute_error(y_test,y_preds):.2f}")
print(f"{color_positive}Train Median Absolute Error {reset_color} -- {median_absolute_error(y_train,y_preds_train):.2f}\n")    

print(f"{color_positive}Test Explained Var {reset_color} -- {explained_variance_score(y_test,y_preds):.2f}")
print(f"{color_positive}Train Explained Var {reset_color} -- {explained_variance_score(y_train,y_preds_train):.2f}\n")    
print('-'*50)
```

Test Mean Squared Error (781.82) is significantly lower than Train Mean Squared Error (409.73). This suggests the model is generalizing well.

```python

```

```python
grid_search = GridSearchCV(estimator=ExtraTreesRegressor(), param_grid=param_grid, 
                        scoring='neg_mean_squared_error', cv=3, verbose=3, n_jobs=1)

grid_search.fit(x_train, y_train)

```

```python
print("Best Parameters:", grid_search.best_params_)
print("Best Negative Mean Squared Error:", grid_search.best_score_)

best_rf_model = grid_search.best_estimator_
y_preds_test = best_rf_model.predict(x_test)
test_mse = mean_squared_error(y_test, y_preds_test)
print("Test Mean Squared Error (Best Model):", test_mse)
```

Suggestions to Fix Underfitting:

Increase Model Complexity: Increase max_depth or reduce min_samples_split.
Add More Trees: Increase n_estimators.

```python
import time
model = ExtraTreesRegressor(max_depth=None, min_samples_leaf= 7, min_samples_split= 5, n_estimators= 200,criterion='squared_error')
name = 'Extra_Tree_Model'
start_time = time.time() 
print(f"{color_positive}{name} - - {model}\n")


model.fit(x_train,y_train)
y_preds=model.predict(x_test)
y_preds_train=model.predict(x_train)
training_time = time.time() - start_time

print(f"{color_positive}Training Time: {training_time:.2f} seconds\n")

print(f"{color_positive}Test Mean Squared Error {reset_color} -- {mean_squared_error(y_test,y_preds):.2f}")
print(f"{color_positive}Train Mean Squared Error {reset_color} -- {mean_squared_error(y_train,y_preds_train):.2f}\n")

print(f"{color_positive}Test Root Mean Squared Error {reset_color} -- {root_mean_squared_error(y_test,y_preds):.2f}")
print(f"{color_positive}Train Root Mean Squared Error {reset_color} -- {root_mean_squared_error(y_train,y_preds_train):.2f}\n")

print(f"{color_positive}Test R2 Score {reset_color} -- {r2_score(y_test,y_preds):.2f}")
print(f"{color_positive}Train R2 Score {reset_color} -- {r2_score(y_train,y_preds_train):.2f}\n")    

print(f"{color_positive}Test Mean Absolute Error {reset_color} -- {mean_absolute_error(y_test,y_preds):.2f}")
print(f"{color_positive}Train Mean Absolute Error {reset_color} -- {mean_absolute_error(y_train,y_preds_train):.2f}\n")    

print(f"{color_positive}Test Median Absolute Error {reset_color} -- {median_absolute_error(y_test,y_preds):.2f}")
print(f"{color_positive}Train Median Absolute Error {reset_color} -- {median_absolute_error(y_train,y_preds_train):.2f}\n")    

print(f"{color_positive}Test Explained Var {reset_color} -- {explained_variance_score(y_test,y_preds):.2f}")
print(f"{color_positive}Train Explained Var {reset_color} -- {explained_variance_score(y_train,y_preds_train):.2f}\n")    
print('-'*50)
```

```python

```

```python
import time

k_value=np.arange(5,8)
knn_cross_val=[]
name = 'KNN'
for k in k_value:
    model = KNeighborsRegressor(n_neighbors=k)
    model.fit(x_train,y_train)
    print(f"K_Value : , {k}    Train Score :  {model.score(x_train,y_train):.4f}     Cross_Val_Score : { cross_val_score(model,x_train,y_train,cv=3).mean():.4f}")
    
    
    
        # cv_score = cross_val_score(model,x_train,y_train, cv=kf)

    
#     y_preds=model.predict(x_test)
#     y_preds_train=model.predict(x_train)
#     training_time = time.time() - start_time


# print(f"{color_positive}Training Time: {training_time:.2f} seconds\n")

# print(f"{color_positive}Test Mean Squared Error {reset_color} -- {mean_squared_error(y_test,y_preds):.2f}")
# print(f"{color_positive}Train Mean Squared Error {reset_color} -- {mean_squared_error(y_train,y_preds_train):.2f}\n")

# print(f"{color_positive}Test Root Mean Squared Error {reset_color} -- {root_mean_squared_error(y_test,y_preds):.2f}")
# print(f"{color_positive}Train Root Mean Squared Error {reset_color} -- {root_mean_squared_error(y_train,y_preds_train):.2f}\n")

# print(f"{color_positive}Test R2 Score {reset_color} -- {r2_score(y_test,y_preds):.2f}")
# print(f"{color_positive}Train R2 Score {reset_color} -- {r2_score(y_train,y_preds_train):.2f}\n")    

# print(f"{color_positive}Test Mean Absolute Error {reset_color} -- {mean_absolute_error(y_test,y_preds):.2f}")
# print(f"{color_positive}Train Mean Absolute Error {reset_color} -- {mean_absolute_error(y_train,y_preds_train):.2f}\n")    

# print(f"{color_positive}Test Median Absolute Error {reset_color} -- {median_absolute_error(y_test,y_preds):.2f}")
# print(f"{color_positive}Train Median Absolute Error {reset_color} -- {median_absolute_error(y_train,y_preds_train):.2f}\n")    

# print(f"{color_positive}Test Explained Var {reset_color} -- {explained_variance_score(y_test,y_preds):.2f}")
# print(f"{color_positive}Train Explained Var {reset_color} -- {explained_variance_score(y_train,y_preds_train):.2f}\n")    
# print('-'*50)
```

![image.png](attachment:image.png)

```python

```

```python

```

![image.png](attachment:image.png)


![image.png](attachment:image.png)


![image.png](attachment:image.png)


![image.png](attachment:image.png)


![image.png](attachment:image.png)


model is performing well but shows slight signs of overfitting. With some regularization and further hyperparameter tuning, you can improve its generalization to the test set.



![image.png](attachment:image.png)

```python

```
