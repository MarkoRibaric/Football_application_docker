import mysql.connector
from flask import Flask, jsonify
import time
import pandas as pd
from sqlalchemy import create_engine, text
import requests
app = Flask(__name__)

# Database connection configuration
db_config = {
    'host': '127.0.0.1',
    'user': 'my_db_user',
    'password': 'S3cret',
    'database': 'my_db',
    'port': 3308
}
connection = mysql.connector.connect(**db_config)
connection.close()

# Load CSV data into a Pandas DataFrame
df = pd.read_csv('2020-2021.csv')

# Select only the first 8 columns
df_subset = df.iloc[:, 1:7]
# Connect to MySQL database using SQLAlchemy
engine = create_engine(f"mysql+mysqlconnector://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}")

# Define a custom table creation SQL statement
create_table_query = """
    CREATE TABLE IF NOT EXISTS football_data (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        Date TEXT,
        Time TEXT,
        HomeTeam TEXT,
        AwayTeam TEXT,
        FTHG BIGINT,
        FTAG BIGINT,
        FTR TEXT
    )
"""

create_teams_table_query = """
    CREATE TABLE IF NOT EXISTS teams (
        id INT PRIMARY KEY,
        position INT,
        team_name VARCHAR(255),
        playedGames INT,
        won INT,
        draw INT,
        lost INT,
        points INT,
        goalsFor INT,
        goalsAgainst INT,
        goalDifference INT
    )
"""


team_mapping = {
    'Arsenal': 'Arsenal FC',
    'Aston Villa': 'Aston Villa FC',
    'Chelsea': 'Chelsea FC',
    'Everton': 'Everton FC',
    'Fulham': 'Fulham FC',
    'Liverpool': 'Liverpool FC',
    'Man City': 'Manchester City FC',
    'Man United': 'Manchester United FC',
    'Tottenham': 'Tottenham Hotspur FC',
    'Wolves': 'Wolverhampton Wanderers FC',
    'Crystal Palace': 'Crystal Palace FC',
    'Sheffield United': 'Sheffield United FC',
    'Brighton': 'Brighton & Hove Albion FC',
    'Brentford': 'Brentford FC',
    'West Ham': 'West Ham United FC',
    'Newcastle': 'Newcastle United FC',
    'Leeds': 'Leeds FC',
    'Burnley': 'Burnley FC'
}


def map_team(team):
    return team_mapping.get(team, team)


try:
    connection = engine.connect()
    connection.execute(text(create_teams_table_query))
    api_url = 'https://api.football-data.org/v4/competitions/PL/standings'
    headers = {'X-Auth-Token': '5981eacaedd4458f9415848e00f97473'}
    response = requests.get(api_url, headers=headers)

    data = response.json()

    for team_info in data["standings"][0]["table"]:
        position = team_info["position"]
        team_id = team_info["team"]["id"]
        team_name = team_info["team"]["name"]
        played_games = team_info["playedGames"]
        team_crest = team_info["team"]["crest"]
        won = team_info["won"]
        draw = team_info["draw"]
        lost = team_info["lost"]
        points = team_info["points"]
        goals_for = team_info["goalsFor"]
        goals_against = team_info["goalsAgainst"]
        goal_difference = team_info["goalDifference"]

        
        df_subset_teams = pd.DataFrame({
            'id': team_id,
            'position': position,
            'team_name': team_name,
            'playedGames': played_games,
            'won': won,
            'draw': draw,
            'lost': lost,
            'points': points,
            'goalsFor': goals_for,
            'goalsAgainst': goals_against,
            'goalDifference': goal_difference
        }, index=[team_id])


        df_subset_teams.reset_index(drop=True, inplace=True)
        df_subset_teams.to_sql('teams', con=engine, if_exists='append', index=False)



except Exception as e:
    # Print an error message if an exception occurs
    print(f"Error: {e}")


try:
    connection = engine.connect()

    connection.execute(text(create_table_query))
    df_subset['HomeTeam'] = df_subset['HomeTeam'].map(map_team)
    df_subset['AwayTeam'] = df_subset['AwayTeam'].map(map_team)
    df_subset.to_sql('football_data', con=engine, if_exists='replace', index=True, method='multi')

    print("Data successfully written to MySQL database.")

except Exception as e:
    # Print an error message if an exception occurs
    print(f"Error: {e}")



finally:
    # Close the database connections
    if connection:
        connection.close()
    engine.dispose()
