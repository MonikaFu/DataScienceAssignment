# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 13:45:40 2018

@author: Monika
"""

import pandas as pd
import os
import numpy as np
#import datetime

# home laptop
os.chdir('C:\dev\ING_assignment')                                              
# work laptop
#os.chdir('C:\\Users\\N91230.LAUNCHER\\Documents\\Private\\ING_assignment')         

# get the data in
sales_data = pd.read_csv('data\store_sales_per_category.csv', sep=';')
distance_data = pd.read_csv('data\store_distances_anonymized.csv')
education_data = pd.read_csv('data\gdata_anonymized.csv')

# investigate the properties of the datasets

# are all weeks in all years present in the dataset?
unique_year_week = sales_data[['year','week']].drop_duplicates().reset_index()

# are all shops in all weeks present in the dataset?
unique_year_week_shop = sales_data[['year','week','store_id']].drop_duplicates().groupby(['year','week']).size()
unique_shop_ids = pd.unique(sales_data['store_id'])

# are the distnaces repeated for store_id2
unique_store_ids1 = distance_data['store_id_1'].drop_duplicates()
unique_store_ids2 = distance_data['store_id_2'].drop_duplicates()
overlapping_ids = unique_store_ids1.isin(unique_store_ids2)

#check for nan values
sales_data.isnull().any().any()
distance_data.isnull().any().any()
education_data.isnull().any().any()
# no nans -> proceed

# create a variable indicating performance of the shop - I chose for weekly 
# sales in order to get as much data to model on as possible
sales_data['weekly_sales']=sales_data.loc[:,'Vodka':'Rum'].sum(axis=1)

sales_data=sales_data.sort_values(['store_id','year','week'])

# add weeks with no sales and fill them with mean per shop
sales_data = sales_data.set_index(['store_id', 'year', 'week'])

for id in unique_shop_ids:
    # get the data for this id
    id_data = sales_data.loc[id]
    # what is the reasonable number of datapoints to model on? I chose 20 
    # to minimize messiness of the dataset - around half of year of data is available
    if id_data.shape[0]>19:
        min_year_week = id_data.index[0]
        max_year_week = id_data.index[-1]
        mean_sales = id_data['weekly_sales'].mean()
        active_weeks_1 = unique_year_week[unique_year_week['year']>min_year_week[0]]
        active_weeks_0 = unique_year_week[(unique_year_week['year']==min_year_week[0]) & (unique_year_week['week']>=min_year_week[1])]
        active_weeks_2 = unique_year_week[(unique_year_week['year']==max_year_week[0]) & (unique_year_week['week']<=max_year_week[1])]
        active_weeks = pd.concat([active_weeks_0, active_weeks_1, active_weeks_2])
        # we fill in at most 25% missing values
        if active_weeks.shape[0]<1.25*id_data.shape[0]:
            tuples = [tuple(x) for x in active_weeks[['year','week']].values]
            new_index = pd.MultiIndex.from_tuples(tuples)
            id_data=id_data.reindex(new_index)
            id_data['weekly_sales']=id_data['weekly_sales'].fillna(mean_sales)
            #add the lagged sales data
            id_data['weekly_sales_lag1']=id_data['weekly_sales'].shift()
            id_data['weekly_sales_lag2']=id_data['weekly_sales'].shift(2)
            id_data['store_id']=id
            id_data.reset_index(inplace=True)
            #id_data.set_index(['store_id','level_0','level_1'],inplace=True)
            if not ('sales_data_fill_mis' in locals()):
                sales_data_fill_mis=id_data
            else:
                sales_data_fill_mis=pd.concat([sales_data_fill_mis,id_data])

sales_data_fill_mis.reset_index(inplace=True)
sales_data_fill_mis.drop(['index'],axis=1,inplace=True)

# add the information on number of universities nearby
sales_data_fill_mis=sales_data_fill_mis.join(education_data.set_index('store_id'),on='store_id')
sales_data_fill_mis.dropna(subset=['weekly_sales_lag1','weekly_sales_lag2'],inplace=True)

final_shop_ids = sales_data_fill_mis['store_id'].drop_duplicates()
all_ids = final_shop_ids.values

import random
random.shuffle(all_ids)

training_ids = all_ids[0:round(len(all_ids)*0.8)]
test_ids = all_ids[(round(len(all_ids)*0.8)+1):-1]

trainig_set = sales_data_fill_mis[sales_data_fill_mis['store_id'].isin(training_ids)]
test_set = sales_data_fill_mis[sales_data_fill_mis['store_id'].isin(test_ids)]

# fit a liner regression model to lagged variables
import statsmodels.api as sm
model = sm.OLS(trainig_set['weekly_sales'], trainig_set[['weekly_sales_lag1','weekly_sales_lag2','num_universities']]).fit()
predictions = model.predict(test_set[['weekly_sales_lag1','weekly_sales_lag2','num_universities']]) # make the predictions by the model

# Print out the statistics
model.summary()

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(list(range(1,test_set.shape[0]+1)),predictions, 'o', label="Predictions")
ax.plot(list(range(1,test_set.shape[0]+1)),test_set['weekly_sales'], 'r-', label="True")
ax.legend(loc="best");

import sklearn.metrics as sklmet

sklmet.mean_absolute_error(test_set['weekly_sales'],predictions)
sklmet.mean_squared_error(test_set['weekly_sales'],predictions)
sklmet.r2_score(test_set['weekly_sales'],predictions)

# count the number of shops within 5 km for each shop id
nr_shops_near=distance_data.groupby(['store_id_1'])['distance'].count()

nr_shops_near2=distance_data.groupby(['store_id_2'])['distance'].count()

#check if there are ids that appear in both store_id1 and 2
nr_shops_near_fin=pd.concat([nr_shops_near, nr_shops_near2])
nr_shops_near_fin=nr_shops_near_fin.to_frame()
nr_shops_near_fin.reset_index(inplace=True)
nr_shops=nr_shops_near_fin.groupby(['index'])['distance'].sum()

# check how are the store_ids1 and 2 overlapping
test2=nr_shops_near_fin.index.values
[test3, test_counts]=np.unique(test2,return_counts=True)

sales_data_fill_mis = sales_data_fill_mis.join(nr_shops,on='store_id')
sales_data_fill_mis.rename(columns={'distance':'nr_shops_near'},inplace=True)
sales_data_fill_mis['nr_shops_near']=sales_data_fill_mis['nr_shops_near'].fillna(0)


