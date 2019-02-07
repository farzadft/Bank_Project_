
# coding: utf-8
#Author: Farzad Fakhari-Tehrani
#Group Num : 2

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot
import seaborn as sns
import statsmodels.formula.api as sm
import statsmodels.api as sma
get_ipython().magic('matplotlib inline')


# In[2]:


wagesAndCountData = pd.read_csv('113 Wages by immigrant status and education UTF-8.csv', header = 0,  encoding="utf-8")
# Statistics Canada suppresses estimates below 1500 as 0. 
# Dropping rows which have value as 0 as those will become outliers eventually
wagesAndCountData = wagesAndCountData[wagesAndCountData['Value'] > 0]
wagesAndCountData.head()


# In[3]:


wagesData = wagesAndCountData[wagesAndCountData['Char'].str.strip().str.startswith('Average weekly')]
del wagesData['Char']
wagesData.head()


# In[4]:


countData = wagesAndCountData[wagesAndCountData['Char'].str.strip().str.startswith('Total employees')]
del countData['Char']
countData.head()


# In[5]:


# Wage analysis by Education Level
educationLevel=['0 - 8  years','High school graduate','Post-secondary certificate or diploma','Bachelor\'s degree', 'Above bachelor\'s degree']
wagesDataByEducation = wagesData[wagesData['Education level'].str.strip().str.startswith(tuple(educationLevel))]
wagesDataByEducation = wagesDataByEducation[wagesDataByEducation['SEX'].str.strip().str.startswith('Both sexes')]
wagesDataByEducation = wagesDataByEducation[wagesDataByEducation['AGE GROUP'].str.strip().str.startswith('15 +')]
wagesDataByEducation = wagesDataByEducation[wagesDataByEducation['Immig'].str.strip().str.endswith('Total')]
wagesDataByEducation = wagesDataByEducation[wagesDataByEducation['TYPE OF WORK'].str.strip().str.endswith('All employees')]
wagesDataByEducation = wagesDataByEducation.drop(['SEX', 'AGE GROUP', 'Immig', 'TYPE OF WORK', 'Year'], axis=1)
wagesDataByEducationGroup = wagesDataByEducation.groupby(['Education level','GEOGRAPHY']).agg({'Value':'mean'}).sort_values("Value")
wagesDataByEducationGroup.rename(columns={'Value': 'Wages per week'}, inplace=True)
wagesDataByEducationGroup


# In[6]:


fig, ax = plt.subplots(figsize=(15,7))
wagesDataByEducationGroup.unstack().plot.bar(figsize=(15,8), ax=ax,title="Wage analysis by Education Level (Ontario & Canada)")
wagesDataByEducationGroup.plot.bar(figsize=(15,8), title="Wage analysis by Education Level")


# In[7]:


wagesDataByEducation = wagesData[wagesData['Education level'].str.strip().str.startswith(tuple(educationLevel))]
wagesDataByEducation = wagesDataByEducation[wagesDataByEducation['SEX'].str.strip().str.startswith('Both sexes')]
wagesDataByEducation = wagesDataByEducation[wagesDataByEducation['AGE GROUP'].str.strip().str.startswith('15 +')]
wagesDataByEducation = wagesDataByEducation[wagesDataByEducation['Immig'].str.strip().str.endswith('Total')]
wagesDataByEducation = wagesDataByEducation[wagesDataByEducation['TYPE OF WORK'].str.strip().str.endswith('All employees')]
wagesDataByEducation = wagesDataByEducation.drop(['SEX', 'AGE GROUP', 'Immig', 'TYPE OF WORK', 'GEOGRAPHY'], axis=1)
wagesDataByEducation = wagesDataByEducation.groupby(['Education level', 'Year']).agg({'Value':'mean'})
wagesDataByEducation.rename(columns={'Value': 'Wage'}, inplace=True)
wagesDataByEducationOnYear = wagesDataByEducation.unstack().transpose()
wagesDataByEducationOnYear


# In[8]:


pd.plotting.scatter_matrix(wagesDataByEducationOnYear, diagonal='hist',marker='x',  figsize=(15, 20), c='red');


# In[9]:


autocorrelation_plot(wagesDataByEducationOnYear)


# In[10]:


# Wage analysis by Immigration seniority
immigrationSeniority=['Very recent immigrants', 'Recent immigrants 5+ years','Recent immigrants, 5+ to 10 years','Established immigrants, 10+ years']
wagesDataByimmigrationSeniority = wagesData[wagesData['Immig'].str.strip().str.startswith(tuple(immigrationSeniority))]
wagesDataByimmigrationSeniority = wagesDataByimmigrationSeniority[wagesDataByimmigrationSeniority['SEX'].str.strip().str.startswith('Both sexes')]
wagesDataByimmigrationSeniority = wagesDataByimmigrationSeniority[wagesDataByimmigrationSeniority['AGE GROUP'].str.strip().str.startswith('15 +')]
wagesDataByimmigrationSeniority = wagesDataByimmigrationSeniority[wagesDataByimmigrationSeniority['Education level'].str.strip().str.startswith('Total')]
wagesDataByimmigrationSeniority = wagesDataByimmigrationSeniority[wagesDataByimmigrationSeniority['TYPE OF WORK'].str.strip().str.endswith('All employees')]
wagesDataByimmigrationSeniority = wagesDataByimmigrationSeniority.drop(['SEX', 'AGE GROUP', 'Education level', 'TYPE OF WORK', 'Year'], axis=1)
wagesDataByimmigrationSeniority = wagesDataByimmigrationSeniority.groupby(['Immig','GEOGRAPHY']).agg({'Value':'mean'}).sort_values("Value")
wagesDataByimmigrationSeniority.rename(columns={'Value': 'Wages per week'}, inplace=True)
wagesDataByimmigrationSeniority


# In[11]:


fig, ax = plt.subplots(figsize=(15,7))
wagesDataByimmigrationSeniority.unstack().plot.bar(figsize=(15,8), ax=ax,title="Wage analysis by Immigration Seniority(Ontario & Canada)")
wagesDataByimmigrationSeniority.plot.bar(figsize=(15,8), title="Wage analysis by Immigration Seniority")


# In[12]:


#Wage analysis by sex
sex =['Men', 'Women']
wagesDataBySex = wagesData[wagesData['SEX'].str.strip().str.startswith(tuple(sex))]
wagesDataBySex = wagesDataBySex[wagesDataBySex['Immig'].str.strip().str.endswith('Total')]
wagesDataBySex = wagesDataBySex[wagesDataBySex['AGE GROUP'].str.strip().str.startswith('15 +')]
wagesDataBySex = wagesDataBySex[wagesDataBySex['Education level'].str.strip().str.startswith('Total')]
wagesDataBySex = wagesDataBySex[wagesDataBySex['TYPE OF WORK'].str.strip().str.endswith('All employees')]
wagesDataBySex = wagesDataBySex.drop(['Immig', 'AGE GROUP', 'Education level', 'TYPE OF WORK', 'Year'], axis=1)
wagesDataBySex = wagesDataBySex.groupby(['SEX','GEOGRAPHY']).agg({'Value':'mean'}).sort_values("Value")
wagesDataBySex.rename(columns={'Value': 'Wages per week'}, inplace=True)
wagesDataBySex


# In[13]:


fig, ax = plt.subplots(figsize=(15,7))
wagesDataBySex.unstack().plot.bar(figsize=(15,8), ax=ax,title="Wage analysis by Sex(Ontario & Canada)")


# In[14]:


#Wage analysis by Ontario and Canada
wagesDataByRegion = wagesData[wagesData['Immig'].str.strip().str.endswith('Total')]
wagesDataByRegion = wagesDataByRegion[wagesDataByRegion['SEX'].str.strip().str.startswith('Both sexes')]
wagesDataByRegion = wagesDataByRegion[wagesDataByRegion['AGE GROUP'].str.strip().str.startswith('15 +')]
wagesDataByRegion = wagesDataByRegion[wagesDataByRegion['Education level'].str.strip().str.startswith('Total')]
wagesDataByRegion = wagesDataByRegion[wagesDataByRegion['TYPE OF WORK'].str.strip().str.endswith('All employees')]
wagesDataByRegion = wagesDataByRegion.drop(['SEX','Immig', 'AGE GROUP', 'Education level', 'TYPE OF WORK', 'Year'], axis=1)
wagesDataByRegion = wagesDataByRegion.groupby(['GEOGRAPHY']).agg({'Value':'mean'}).sort_values("Value")
wagesDataByRegion.rename(columns={'Value': 'Wages per week'}, inplace=True)
wagesDataByRegion


# In[15]:


wagesDataByRegion.plot.bar(figsize=(15,8), title="Wage analysis by Region")


# In[16]:


#Wage analysis by type of work 
typeOfWork =['Full-time', 'Part-time']
wagesDataByTypeOfWork = wagesData[wagesData['TYPE OF WORK'].str.strip().str.startswith(tuple(typeOfWork))]
wagesDataByTypeOfWork = wagesDataByTypeOfWork[wagesDataByTypeOfWork['Immig'].str.strip().str.endswith('Total')]
wagesDataByTypeOfWork = wagesDataByTypeOfWork[wagesDataByTypeOfWork['AGE GROUP'].str.strip().str.startswith('15 +')]
wagesDataByTypeOfWork = wagesDataByTypeOfWork[wagesDataByTypeOfWork['Education level'].str.strip().str.startswith('Total')]
wagesDataByTypeOfWork = wagesDataByTypeOfWork[wagesDataByTypeOfWork['SEX'].str.strip().str.endswith('Both sexes')]
wagesDataByTypeOfWork = wagesDataByTypeOfWork.drop(['Immig', 'AGE GROUP', 'Education level', 'SEX', 'Year'], axis=1)
wagesDataByTypeOfWork = wagesDataByTypeOfWork.groupby(['TYPE OF WORK','GEOGRAPHY']).agg({'Value':'mean'}).sort_values("Value")
wagesDataByTypeOfWork.rename(columns={'Value': 'Wages per week'}, inplace=True)
wagesDataByTypeOfWork


# In[17]:


fig, ax = plt.subplots(figsize=(15,7))
wagesDataByTypeOfWork.unstack().plot.bar(figsize=(15,8), ax=ax,title="Wage analysis by Type Of Work")


# In[18]:


# Run a regression about wage rate, with variables of education level, age, sex, geography, immigrant status and type of work.
ageMapping = {'25 - 34':30,'25 - 54':35,'25 - 64':45}
immigMapping = {'Very recent immigrants, 5 years or less':2.5,
                'Recent immigrants 5+ years':5,
                'Recent immigrants, 5+ to 10 years':7.5,
                'Established immigrants, 10+ years':15,
                'Non-landed immigrants':0,
                'Born':'AGE GROUP'}
educationMapping = {'0 - 8  years':4,
                    'Above bachelor\'s degree':19,
                    'Bachelor\'s degree':16,
                    'High school graduate':12,
                    'Post-secondary certificate or diploma':14,
                    'University degree':16,
                    'Some high school':12,
                    'Some post-secondary':14}
typeOfWorkMapping = {'Full-time':1,'Part-time':0}
                                                                                      
wagesRegression = wagesData[wagesData['GEOGRAPHY'].str.strip().str.endswith('Ontario')]
wagesRegression = wagesRegression[wagesRegression['SEX'].str.strip().str.startswith(tuple(sex))]
wagesRegression = wagesRegression[wagesRegression['AGE GROUP'].str.strip().str.startswith(tuple(ageMapping.keys()))]
wagesRegression = wagesRegression[wagesRegression['Immig'].str.strip().str.startswith(tuple(immigMapping.keys()))]
wagesRegression = wagesRegression[wagesRegression['Education level'].str.strip().str.startswith(tuple(educationMapping.keys()))]
wagesRegression = wagesRegression[wagesRegression['TYPE OF WORK'].str.strip().str.startswith(tuple(typeOfWorkMapping.keys()))]

wagesRegression['SEX'] = wagesRegression.apply(lambda row: 0 if row['SEX'].strip()=="Men" else 1, axis=1)
wagesRegression['AGE GROUP'] = wagesRegression.apply(lambda row: ageMapping[row['AGE GROUP'].strip()], axis=1)
wagesRegression['Immig'] = wagesRegression.apply(lambda row: immigMapping[row['Immig'].strip()] if  not row['Immig'].strip().startswith('Born') else row['AGE GROUP'], axis=1)
wagesRegression['Education level'] = wagesRegression.apply(lambda row: educationMapping[row['Education level'].strip()], axis=1)
wagesRegression['TYPE OF WORK'] = wagesRegression.apply(lambda row: typeOfWorkMapping[row['TYPE OF WORK'].strip()], axis=1)
del wagesRegression['GEOGRAPHY']
wagesRegression.head()


# In[19]:


Y = wagesRegression.Value
X = wagesRegression[['Year','SEX', 'AGE GROUP', 'Immig','Education level', 'TYPE OF WORK']]
X["constant"] = 1
result = sm.OLS( Y, X ).fit()
result.summary()


# In[20]:


fig, ax = plt.subplots(figsize=(15,7))
fig = sma.graphics.plot_fit(result, 0, ax=ax)
ax.set_ylabel("Wages")
ax.set_xlabel("Year")
ax.set_title("Linear Regression")
plt.show()


# In[28]:


wagesDataByYear = wagesData.loc[:,["Year", "Value"]].groupby("Year" ).agg({'Value':'mean'})
wagesDataByYear.rename(columns={'Value': 'Wages per week'}, inplace=True)
wagesDataByYear


# In[37]:


wagesDataByYear.plot(kind='LINE')

