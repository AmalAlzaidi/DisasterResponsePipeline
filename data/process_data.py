import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
   

   '''
    Load the messages and categories datasets and merge the two datasets
    
    INPUT - messages_filepath - file path of disaster messages dataset
            categories_filepath - file path of categories dataset
    OUTPUT - 
            df - merged data frame from the messages and categories datasets
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge( messages, categories,  on = 'id')
    
    return df

def clean_data(df):
    
    '''
    split the categories column into separate, rename columns, convert values to binary, and drop duplicates.
    INPUT - 
           df - merged data frame from the messages and categories datasets
    OUTPUT - 
            df - cleaned data frame
    '''
    categories = pd.DataFrame(df['categories'].str.split(';', expand = True))
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    # drop the original categories column from `df`
    df.drop('categories', axis= 1, inplace = True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis = 1)
    # drop duplicates
    df.drop_duplicates(inplace = True)
    
    return df
    
    


def save_data(df, database_filename):
    
    
    '''
    Store the dataset in a SQLite database
    INPUT - 
           df - cleaned data frame
           database_filename - database name
    OUTPUT - 
            SQLite database
            
    '''
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('messages', engine, index=False)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()