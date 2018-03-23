import pandas as pd
import os
import numpy as np

## Set the working directory 

# home laptop
#os.chdir('C:\dev\ING_assignment')                                              
# work laptop
os.chdir('C:\\Users\\N91230.LAUNCHER\\Documents\\Private\\ING_assignment')         

# get the data in
sales_data = pd.read_csv('data\store_sales_per_category.csv', sep=';')
distance_data = pd.read_csv('data\store_distances_anonymized.csv')
education_data = pd.read_csv('data\gdata_anonymized.csv')

## Investigate the properties of the datasets

# are all weeks in all years present in the dataset?
unique_year_week = sales_data[['year','week']].drop_duplicates().reset_index()

# are all shops in all weeks present in the dataset?
unique_year_week_shop = sales_data[['year','week','store_id']].drop_duplicates().groupby(['year','week']).size()
unique_shop_ids = pd.unique(sales_data['store_id'])

#check for nan values
sales_data.isnull().any().any()
distance_data.isnull().any().any()
education_data.isnull().any().any()
# no nans -> proceed

## Prepare for modelling

# create a variable indicating performance of the shop - I chose for weekly 
# sales in order to get as much data to model on as possible
sales_data['weekly_sales']=sales_data.loc[:,'Vodka':'Rum'].sum(axis=1)

# sort values by store_id and year to see how consistent are the times series per store_id
sales_data=sales_data.sort_values(['store_id','year','week'])

# set the multilevel index  to store_id and week
sales_data = sales_data.set_index(['store_id', 'year', 'week'])

# create the modelling dataset - perform operations and checks for each store_id and add (or not)
# their data to the modelling dataset
for id in unique_shop_ids:
    # get the data for this id
    id_data = sales_data.loc[id]
    # what is the reasonable number of datapoints to model on? I chose 20 
    # to minimize messiness of the dataset - at least around half year of data is available
    if id_data.shape[0]>19:
        # find the start and end week of the time series     
        min_year_week = id_data.index[0]
        max_year_week = id_data.index[-1]
        # calculate the mean weekly sales to fill the blanks
        mean_sales = id_data['weekly_sales'].mean()
        # find the weeks that are within the start and end date of the time series of weekly sales for this store
        active_weeks_1 = unique_year_week[unique_year_week['year']>min_year_week[0]]
        active_weeks_0 = unique_year_week[(unique_year_week['year']==min_year_week[0]) & (unique_year_week['week']>=min_year_week[1])]
        active_weeks_2 = unique_year_week[(unique_year_week['year']==max_year_week[0]) & (unique_year_week['week']<=max_year_week[1])]
        active_weeks = pd.concat([active_weeks_0, active_weeks_1, active_weeks_2])
        # I fill in at most 25% missing values - otherwise don't use that store for modelling
        if active_weeks.shape[0]<1.25*id_data.shape[0]:
            # create a new index with all weeks between start and end date in it            
            tuples = [tuple(x) for x in active_weeks[['year','week']].values]
            new_index = pd.MultiIndex.from_tuples(tuples)
            # add the missing dates to the data
            id_data=id_data.reindex(new_index)
            # fill missing data with mean weekly sales for this store
            id_data['weekly_sales']=id_data['weekly_sales'].fillna(mean_sales)
            # add the lagged sales data and id
            id_data['weekly_sales_lag1']=id_data['weekly_sales'].shift()
            id_data['weekly_sales_lag2']=id_data['weekly_sales'].shift(2)
            id_data['store_id']=id
            id_data.reset_index(inplace=True)
            # append the data of this store in the modelling dataset 
            if not ('sales_data_fill_mis' in locals()):
                sales_data_fill_mis=id_data
            else:
                sales_data_fill_mis=pd.concat([sales_data_fill_mis,id_data])

# reset the index so that it is unique per row
sales_data_fill_mis.reset_index(inplace=True)
sales_data_fill_mis.drop(['index'],axis=1,inplace=True)

# add the information on number of universities nearby
sales_data_fill_mis=sales_data_fill_mis.join(education_data.set_index('store_id'),on='store_id')
sales_data_fill_mis.dropna(subset=['weekly_sales_lag1','weekly_sales_lag2'],inplace=True)

## Prepare the trainig and testing datasets

# get all the stores that are selected to be used for modelling
final_shop_ids = sales_data_fill_mis['store_id'].drop_duplicates()
all_ids = final_shop_ids.values

# uncomment below part if you want to get new split into training and testing datasets
#import random
## randomly shuffle the ids
#random.shuffle(all_ids)
#
## pick 80% of the stores (randomly) for training
#training_ids = all_ids[0:round(len(all_ids)*0.8)]
## the remaining 20% of the stores will be used for testing
#test_ids = all_ids[(round(len(all_ids)*0.8)+1):-1]
#
#np.save('training_ids.npy',training_ids)
#np.save('test_ids.npy',test_ids)

# load the training and test store ids and create the training and test datasets
training_ids = np.load('training_ids.npy')
test_ids = np.load('test_ids.npy')

trainig_set = sales_data_fill_mis[sales_data_fill_mis['store_id'].isin(training_ids)]
test_set = sales_data_fill_mis[sales_data_fill_mis['store_id'].isin(test_ids)]

## Modelling
import statsmodels.api as sm

# fit a liner regression model to lagged variables and the number of universities
model = sm.OLS(trainig_set['weekly_sales'], trainig_set[['weekly_sales_lag1','weekly_sales_lag2','num_universities']]).fit()
# make the predictions on the test dataset
predictions = model.predict(test_set[['weekly_sales_lag1','weekly_sales_lag2','num_universities']]) 

# Print out the statistics
model.summary()

# plot the preditions compared with actual data
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(list(range(1,test_set.shape[0]+1)),predictions, 'o', label="Predictions")
ax.plot(list(range(1,test_set.shape[0]+1)),test_set['weekly_sales'], 'r-', label="True")
ax.legend(loc="best");

import sklearn.metrics as sklmet

# calculate performance metrics of the model
sklmet.mean_absolute_error(test_set['weekly_sales'],predictions)
sklmet.mean_squared_error(test_set['weekly_sales'],predictions)
sklmet.r2_score(test_set['weekly_sales'],predictions)

###############################################################################

## TO DO:
# test if adding the information from the distance dataset improves the performance

###############################################################################

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


