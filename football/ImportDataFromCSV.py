from cassandra.cluster import Cluster
from cassandra.query import SimpleStatement
from flask import Flask
import pandas as pd
from pandas.errors import ParserError

cluster = Cluster(['127.0.0.1'], port=9042)
session = cluster.connect()

# Create keyspace if not exists
session.execute("CREATE KEYSPACE IF NOT EXISTS football_data WITH replication = {'class': 'SimpleStrategy', 'replication_factor': '1'}")

# Use the keyspace
session.set_keyspace('football_data')

# Create table if not exists
create_table_query = """
    CREATE TABLE IF NOT EXISTS football_data (
        id UUID PRIMARY KEY,
        Date TEXT,
        Time TEXT,
        HomeTeam TEXT,
        AwayTeam TEXT,
        FTHG INT,
        FTAG INT,
        FTR TEXT,
        HShots INT,
        AShots INT,
        HSTarget INT,
        ASTarget INT,
        HY INT,
        AY INT,
        HR INT,
        AR INT
    )
"""
session.execute(create_table_query)

try:
    required_columns = ['Date', 'Time', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HS', 'AS', 'HST', 'AST', 'HY', 'AY', 'HR', 'AR']

    for i in range(1, 330):
        try:
            df = pd.read_csv(f'csv_files/{i}.csv', encoding='latin1')
        except ParserError as pe:
            print(f"Skipping file {i}.csv due to parsing error: {pe}")
            continue
        if set(required_columns).issubset(df.columns):
            new_df = df[['Date', 'Time', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HS', 'AS', 'HST', 'AST', 'HY', 'AY', 'HR', 'AR']]
            
            df_selected = new_df.rename(columns={'HS': 'HShots', 'AS': 'AShots', 'HST': 'HSTarget', 'AST': 'ASTarget'})
            
            # Try converting relevant columns to integers, and skip rows with non-finite values
            try:
                integer_columns = ['FTHG', 'FTAG', 'HShots', 'AShots', 'HSTarget', 'ASTarget', 'HY', 'AY', 'HR', 'AR']
                df_selected[integer_columns] = df_selected[integer_columns].astype(int)
            except ValueError as ve:
                print(f"Skipping rows with non-finite values in file {i}.csv")
                continue
            
            for _, row in df_selected.iterrows():
                insert_query = """
                    INSERT INTO football_data (id, Date, Time, HomeTeam, AwayTeam, FTHG, FTAG, FTR, HShots, AShots, HSTarget, ASTarget, HY, AY, HR, AR)
                    VALUES (uuid(), %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                session.execute(insert_query, (row['Date'], row['Time'], row['HomeTeam'], row['AwayTeam'], row['FTHG'], row['FTAG'], row['FTR'], row['HShots'], row['AShots'], row['HSTarget'], row['ASTarget'], row['HY'], row['AY'], row['HR'], row['AR']))
                
            print(f"Data from file {i}.csv successfully written to Cassandra database.")
        else:
            print(f"File {i}.csv does not contain all the required columns. Skipping.")
except Exception as e:
    print(f"Error: {e}")

finally:
    cluster.shutdown()
