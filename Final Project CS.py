# Final CS Project - Vedant Nilabh, Ahmad Saeed, Isabel Finkbeiner
# making data frames
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df1 = pd.read_csv('Dallas_Market_Value_Analysis.csv')
pd.set_option('display.max_columns', None)
print(df1.head())
df2 = pd.read_csv('NH_DallasCountyBlockGroups2010WithAppraisalRollups.csv')
print(df2.head())


# cleaning data series for merging and merging into one dataframe
df1.geoid = df1.geoid.str.replace('a', '')
df1.geoid = df1.geoid.str.replace('b', '').astype(int)
#print(df1.dtypes)
#print(df2.dtypes)
df = pd.merge(df1, df2, left_on="geoid", right_on="BLOCKGROUP")
#print(df.head())
##print(df.dtypes)
#print(df[df.isnull().any(axis=1)])

# choosing regression variables
X = df[['MdSalesPr', 'PFclOO1517', 'PPermNCUnt', 'PCVLnRsPr', 'PRehabPerm', 'PPubSubAll', 'POPDENS', 'TotalVal']]
y = df['P_MINORITY']



# demographic percentages
##print(np.mean(df['WHITE']) / np.mean(df['POP']))
##print(np.mean(df['HISPANIC']) / np.mean(df['POP']))
##print(np.mean(df['BLACK']) / np.mean(df['POP']))
##print(np.mean(df['MINORITY']) / np.mean(df['POP']))

# normalizing data and running linear regression
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_norm = scaler.fit_transform(X)
#print(X.head())

from sklearn.linear_model import LinearRegression
model = LinearRegression(fit_intercept=True)
model.fit(X_norm,y)

print("Model intercept:", model.intercept_)
coefResults = list(zip(X.columns, model.coef_))
for coefResult in coefResults:
    print(str(coefResult[0]).ljust(30)," ",str(coefResult[1]).rjust(25))

# 3rd dataframe for budget data
df3 = pd.read_csv('CoD_Expenses_Budget_vs_Actual.csv')
df3 = df3.groupby(['APPROPRIATION'])['CURRENT BUDGET'].sum()
pd.set_option('display.max_rows', None)
df3 = df3.sort_values(ascending=False)
print(df3)

# making bar chart for budget data by category
df3 = df3.head(36)
x = df3.plot(kind='bar')
x.set_title('Different Sector Expenses')
x.set_xlabel('Sector')
x.set_ylabel('Expenses')
plt.show()

