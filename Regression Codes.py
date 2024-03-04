import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

os.chdir('C:/Users/Nelson/PycharmProjects/NIIP forecast project')
if not os.path.exists('Data'):
    os.mkdir('Data')

# Creating Dataframes from csv files for OECD GDP, World GDP, and US NIIP Price Valuation effects
df_OECD = pd.read_csv('Data/OECD GDP.csv')
df_WORLD_RAW = pd.read_csv('Data/WORLD GDP.csv')
df_NIIP_RAW = pd.read_csv('Data/US NIIP PRICE VALUATION CHANGES.csv')
df_US_GDP = pd.read_csv('Data/US GDP.csv')
df_US_CPI = pd.read_csv('Data/US CPI.csv')
df_US_NIIP = pd.read_csv('Data/US NIIP.csv')

# Setting up OECD GDP over 1992-2022
df_OECD = df_OECD[df_OECD['LOCATION'] == 'OECD'][['TIME', 'Value']]
df_OECD = df_OECD.rename(columns={'Value': 'OECD GDP'})
df_OECD['OECD GDP'] = df_OECD['OECD GDP']*1000000

# Setting up World GDP over 1992-2022
years = df_OECD['TIME'].unique()
df_WORLD = pd.DataFrame()
df_WORLD['TIME'] = years
df_WORLD['WORLD GDP'] = df_WORLD_RAW[df_WORLD_RAW['Country Code'] == 'WLD'][df_WORLD_RAW.columns[4:]].T.values
df_WORLD['WORLD GDP'] = pd.to_numeric(df_WORLD['WORLD GDP'], errors='coerce')

# Setting up valuation effects from price on US NIIP 2003-2023
df_NIIP = pd.DataFrame()
years, sums = years[11:], []
for y in years:
    sums.append(df_NIIP_RAW[f'Price changes {y}'].sum())
df_NIIP['TIME'] = years
df_NIIP['Price Valuation on NIIP'] = sums
df_NIIP['Price Valuation on NIIP'] = df_NIIP['Price Valuation on NIIP']*1000000

# Creating dataframe with all values
df_ALL = pd.merge(df_OECD, df_WORLD, on='TIME', how='inner')
df_ALL = df_ALL.merge(df_NIIP, on='TIME', how='inner')
df_ALL['NON-OECD GDP'] = df_ALL['WORLD GDP'] - df_ALL['OECD GDP']

# Getting % of change in World GDP caused by NON-OECD countries
sums, years = [], years[:-1]
df_ALL.set_index('TIME', inplace=True)
for y in years:
    sums.append((1-(df_ALL.loc[y+1, 'OECD GDP']-df_ALL.loc[y, 'OECD GDP'])/(df_ALL.loc[y+1, 'WORLD GDP']-df_ALL.loc[y, 'WORLD GDP'])))
sums.insert(0, np.nan)
df_ALL['NON OECD % OF WORLD GDP CHANGE'] = sums
df_ALL = df_ALL.reset_index(drop=False)
if not os.path.exists('Regression Data.xlsx'):
    df_ALL.to_excel('Regression Data.xlsx', index=False)

# ols
X = df_ALL['NON OECD % OF WORLD GDP CHANGE'].values[1:]
Y = df_ALL['Price Valuation on NIIP'].values[1:]
X = sm.add_constant(X)
model = sm.OLS(Y, X).fit()
print(model.summary())

# Setting up US GDP and CPI tables
df_GDP_CPI = pd.DataFrame()
df_US_GDP.set_index('Line', inplace=True)
df_US_CPI.set_index('Label', inplace=True)
years, gdp, cpi = df_US_GDP.columns.tolist(), [], []
for y in years:
    gdp.append(df_US_GDP.loc[1, y])
    cpi.append(df_US_CPI.loc[f'{y} Jan', 'Value'])
df_GDP_CPI['US GDP'] = gdp
df_GDP_CPI['US CPI'] = cpi
df_GDP_CPI['US CPI'] = pd.to_numeric(df_GDP_CPI['US CPI'], errors='coerce')

# ols
X = df_GDP_CPI['US GDP'].values
Y = df_GDP_CPI['US CPI'].values
X = sm.add_constant(X)
model = sm.OLS(Y, X).fit()
print(model.summary())

years, sums = df_US_NIIP.columns[29:], []
df_US_NIIP.set_index('Line', inplace=True)
for y in years:
    sums.append(df_US_NIIP.loc[1, y])
df_ALL['US NIIP'] = sums
df_ALL['US NIIP'] = df_ALL['US NIIP']*1000000
df_ALL['VP as % of NIIP'] = df_ALL['Price Valuation on NIIP']/df_ALL['US NIIP']
df_GDP_CPI['US GDP'] = df_GDP_CPI['US GDP']*1000000
gdp = df_GDP_CPI['US GDP'][11:].values
df_ALL['VP as % GDP'] = df_ALL['Price Valuation on NIIP']/gdp

fig, ax = plt.subplots()
ax.plot(years, df_ALL['VP as % GDP'].values, marker='o', linestyle='-', color='b', label='VP as % of US GDP')
ax.set_xlabel('Year')
ax.set_ylabel('Values')
ax.set_title('Valuation Effect as percentage of GDP')
ax.legend()

plt.grid()
plt.show()
print('hi!')