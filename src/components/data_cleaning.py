import os
import sys
import numpy as np
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from dataclasses import dataclass


@dataclass
class DataCleaningConfig:
    initial_data_path: str = os.path.join('cleaned_data', "data.csv")

class DataCleaning:
    def __init__(self):
        self.cleaning_config = DataCleaningConfig()

    def initiate_data_cleaning(self):
        logging.info("Entered the data loading and data cleaning")
        try:
           df = pd.read_csv('notebook\data\Bengaluru_House_Data.csv')
           logging.info('Exported or read the datasets as dataframe')
           os.makedirs(os.path.dirname(self.cleaning_config.initial_data_path), exist_ok=True)
           
           
           df = df.drop(['area_type','availability','society','balcony'], axis=1)
           df = df.dropna()
           df['bhk'] = df['size'].apply(lambda x: int(x.split(' ')[0]))
           df['total_sqft'] = df['total_sqft'].apply(self.convert_range_to_num)

           df['price_per_sqft'] = df['price']*100000/df['total_sqft']

           df = self.reducing_location(df)

           df = df[~(df['total_sqft']/df['bhk']<300)]
           df = self.remove_pps_ouliers(df)
           df = self.remove_bhk_outliers(df)
           df = df[df.bath < df.bhk+2]
           df = df.drop(['size','price_per_sqft'], axis=1)

           df.to_csv(self.cleaning_config.initial_data_path, index=False, header=True)


        except Exception as e:
            raise CustomException(e, sys)   
           
    def reducing_location(self, df):
        df['location'] = df['location'].apply(lambda x: x.strip())
        location_stats = df.groupby('location')['location'].agg('count').sort_values(ascending = False)
        location_stats_less_than_10 = location_stats[location_stats <=10]
        
        df['location']= df['location'].apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
        df = df[~(df['total_sqft']/df['bhk']<300)]
        return df
    
    def remove_pps_ouliers(self, df):
        df_out = pd.DataFrame()
        for key , subdf in df.groupby('location'):
            m = np.mean(subdf.price_per_sqft)
            st = np.std(subdf.price_per_sqft)
            reduced_df = subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]
            df_out = pd.concat([df_out,reduced_df], ignore_index =True)
        return df_out

    def remove_bhk_outliers(self, df):
        exclude_indices = np.array([])
        for location, location_df in df.groupby('location'):
            bhk_stats = {}
            for bhk, bhk_df in location_df.groupby('bhk'):
                bhk_stats[bhk]={
                    'mean': np.mean(bhk_df.price_per_sqft),
                    'std': np.std(bhk_df.price_per_sqft),
                    'count': bhk_df.shape[0]
                }
            for bhk,bhk_df in location_df.groupby('bhk'):
                stats = bhk_stats.get(bhk-1)
                if stats and stats['count']>5:
                    exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft < (stats['mean'])].index.values)
        return df.drop(exclude_indices,axis ='index')
        
    def convert_range_to_num(self,x):
        tokens = x.split('-')
        if len(tokens) == 2:
            return (float(tokens[0])+ float(tokens[1]))/2
        try:
            return float(x)
        except:
            return None
        
# if __name__ == "__main__":
#     obj = DataCleaning()
#     obj.initiate_data_cleaning()