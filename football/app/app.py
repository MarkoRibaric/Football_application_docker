from flask import Flask, render_template, request, redirect, url_for
from sqlalchemy import create_engine, text, exc, inspect
import pandas as pd
import requests
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from datetime import datetime
import dill
import sklearn
import ast
import shap
import logging
from flask import session, jsonify
logging.basicConfig(level=logging.DEBUG)
from shap import Explanation
app = Flask(__name__)
model_filename = 'random_forest_model.pkl'
with open(model_filename, 'rb') as file:
    model = dill.load(file)

app.secret_key = 'yazjazjazjaz'

db_config = {
    'host': "database_master",
    'user': 'my_db_user',
    'password': 'S3cret',
    'database': 'my_db',
    'port': 3306
}

engine = create_engine(f"mysql+mysqlconnector://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}")


def get_team_list():
    try:
        with engine.connect() as connection:
            teams_query = text("SELECT id, position, team_name, playedGames, won, draw, lost, points, goalsFor, goalsAgainst, goalDifference FROM teams ORDER BY position")
            teams_df = pd.read_sql_query(teams_query, connection)
        return teams_df
    except exc.SQLAlchemyError as e:
        print(f"Error fetching team list: {e}")
        return None


@app.route('/')
def display_data():
    prediction = request.args.get('prediction')
    input_data=request.args.get('input_data')
    custom_game_json = request.args.get('custom_game_json')
    plot_base64_7 = session.get('plot_base64_7', None)
    if (input_data):
        input_data = [int(i) for i in input_data.split(",")]
    inspector = inspect(engine)
    print(input_data)
  
    if 'football_data' in inspector.get_table_names():
        query = "SELECT * FROM football_data"
        df = pd.read_sql_query(query, engine)
        column_order = ['index', 'Date', 'Time', 'HomeTeam', 'FTHG', 'FTAG', 'AwayTeam']
        df = df[column_order]
        df = df[::-1]
        table_html = df.to_html(index=False, classes='table table-striped')

        # subplots
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
        
        # HISTOGRAM ZA HOME GOALS
        axes[0, 0].hist(df['FTHG'], bins=range(df['FTHG'].max() + 2), align='left', color='blue', edgecolor='black')
        axes[0, 0].set_title('Home Goals Histogram')
        axes[0, 0].set_xlabel('Goals')
        axes[0, 0].set_ylabel('Frequency')

        #  HISTOGRAM ZA AWAY GOALS
        axes[0, 1].hist(df['FTAG'], bins=range(df['FTAG'].max() + 2), align='left', color='green', edgecolor='black')
        axes[0, 1].set_title('Away Goals Histogram')
        axes[0, 1].set_xlabel('Goals')
        axes[0, 1].set_ylabel('Frequency')

        # TOP 7 IGRAJUCIH VREMENA
        time_distribution = df['Time'].value_counts()
        top_7_times = time_distribution.head(7)

        labels = top_7_times.index
        sizes = top_7_times.values


        axes[1, 0].bar(labels, sizes, color=['red', 'orange', 'yellow', 'green', 'blue'])
        axes[1, 0].set_title('Distribution of Games by Time (Top 7) (GMT)')
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Number of Games')
        for i, v in enumerate(sizes):
            axes[1, 0].text(i, v + 0.1, str(v), ha='center', va='bottom', fontsize=10)

        
        # GODINE
        df2 = df
        df2['Year'] = pd.to_datetime(df2['Date'], format='%d/%m/%Y').dt.year

        games_per_year = df2['Year'].value_counts().sort_index()
        games_per_year.plot(kind='bar', ax=axes[1, 1], color='skyblue')
        axes[1, 1].set_title('Games per Year')
        axes[1, 1].set_xlabel('Year')
        axes[1, 1].set_ylabel('Number of Games')

        #spremanje
        plt.tight_layout()
        plot_image = BytesIO()
        plt.savefig(plot_image, format='png')
        plot_image.seek(0)
        plot_base64 = base64.b64encode(plot_image.getvalue()).decode('utf-8')


        # INDIVIDUALNI TIMOVI

        fig2, axes2 = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
        selected_team_id = request.args.get('team_id')
        if selected_team_id:
            df_subset = df[(df['HomeTeam'] == selected_team_id) | (df['AwayTeam'] == selected_team_id)]
        else:
            df_subset = df

        team_home = df_subset[df_subset['HomeTeam'] == selected_team_id]
        goal_distribution = team_home['FTHG'].value_counts()
        axes2[0, 0].pie(goal_distribution, labels=goal_distribution.index, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
        axes2[0, 0].set_title(f'Goal Distribution for {selected_team_id} Home Games')

        
        team_home = df_subset[df_subset['AwayTeam'] == selected_team_id]
        goal_distribution = team_home['FTAG'].value_counts()
        axes2[0, 1].pie(goal_distribution, labels=goal_distribution.index, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
        axes2[0, 1].set_title(f'Goal Distribution for {selected_team_id} Away Games')

        time_distribution = df_subset['Time'].value_counts()
        top_7_times = time_distribution.head(7)

        labels = top_7_times.index
        sizes = top_7_times.values

        axes2[1, 0].bar(labels, sizes, color=['red', 'orange', 'yellow', 'green', 'blue'])
        axes2[1, 0].set_title('Distribution of Games by Time (Top 7) (GMT)')
        axes2[1, 0].set_xlabel('Time')
        axes2[1, 0].set_ylabel('Number of Games')

        for i, v in enumerate(sizes):
            axes2[1, 0].text(i, v + 0.1, str(v), ha='center', va='bottom', fontsize=10)

        df_subset['Date'] = pd.to_datetime(df_subset['Date'], format='%d/%m/%Y')

        df_subset['Month'] = df_subset['Date'].dt.month_name()
        monthly_distribution = df_subset['Month'].value_counts().sort_index()
        

        monthly_distribution.plot(kind='bar', ax=axes2[1, 1], color='purple')
        axes2[1, 1].set_title(f'Games per Month for {selected_team_id}')
        axes2[1, 1].set_xlabel('Month')
        axes2[1, 1].set_ylabel('Number of Games')


        for i, v in enumerate(monthly_distribution):
            axes2[1, 1].text(i, v + 0.1, str(v), ha='center', va='bottom', fontsize=10)
        


        if custom_game_json:
            custom_game_dict = ast.literal_eval(custom_game_json) 

            custom_game_df = pd.DataFrame([custom_game_dict])

            # Make sure the columns match the order expected by the model
            custom_game_df = custom_game_df[['hshots', 'ashots', 'hstarget', 'astarget', 'hy', 'ay', 'hr', 'ar']]

            predicted_class = model.predict(custom_game_df)[0]
            predicted_class_proba = model.predict_proba(custom_game_df)[0]

            explainer = shap.TreeExplainer(model)
            shap_values_custom = explainer(custom_game_df)

            exp_custom = shap.Explanation(
                shap_values_custom.values[:, :, predicted_class],
                shap_values_custom.base_values[:, predicted_class],
                data=custom_game_df.values,
                feature_names=custom_game_df.columns
            )

            plt.figure(figsize=(12, 6))
            shap.waterfall_plot(exp_custom[0], show=False)

            plot_image = BytesIO()
            plt.savefig(plot_image, format='png', bbox_inches='tight')
            plt.close()

            plot_base64_7 = base64.b64encode(plot_image.getvalue()).decode('utf-8')


        else:
            custom_game_df = None
            
                                
        plt.tight_layout()

        plot_image = BytesIO()
        plt.savefig(plot_image, format='png', bbox_inches='tight', dpi=100)
        plt.close()
        plot_base64_2 = base64.b64encode(plot_image.getvalue()).decode('utf-8')







        teams_df = get_team_list()

    if df is not None:
        return render_template('index.html', df=df, table_html=table_html, plot_base64=plot_base64, teams_df=teams_df, plot_base64_2=plot_base64_2, prediction=prediction, input_data=input_data, plot_base64_7=plot_base64_7)

    else:
        return render_template('index.html', table_html='', plot_base64='', plot_base64_2='', prediction='', input_data='')

@app.route('/add_data', methods=['POST'])
def add_data():
    data = {
        'Date': request.form['date'],
        'Time': request.form['time'],
        'HomeTeam': request.form['home_team'],
        'FTHG': request.form['fthg'],
        'FTAG': request.form['ftag'],
        'AwayTeam': request.form['away_team']
    }
    #date
    try:
        date_string = request.form['date']
        date_object = datetime.strptime(date_string, '%d/%m/%Y')

        formatted_date = date_object.strftime('%d/%m/%Y')
        data['Date'] = formatted_date
    except ValueError:
        return redirect(url_for('display_data'))


    try:
        time_string = request.form['time']
        time_object = datetime.strptime(time_string, '%H:%M')

        formatted_time = time_object.strftime('%H:%M')

        data['Time'] = formatted_time

    except ValueError:
        return redirect(url_for('handle_invalid_time'))

    duplicate_check_query = f"SELECT * FROM football_data WHERE Date = '{data['Date']}' AND HomeTeam = '{data['HomeTeam']}' AND AwayTeam = '{data['AwayTeam']}'"
    existing_data = pd.read_sql_query(duplicate_check_query, engine)
    duplicate_check_query2 = f"SELECT * FROM football_data WHERE Date = '{data['Date']}' AND HomeTeam = '{data['AwayTeam']}' AND AwayTeam = '{data['HomeTeam']}'"
    existing_data2 = pd.read_sql_query(duplicate_check_query2, engine)

    if not existing_data.empty or not existing_data2.empty:
        return redirect(url_for('display_data'))

    current_max_index = pd.read_sql_query("SELECT MAX(`index`) FROM football_data", con=engine).iloc[0, 0]
    new_index = current_max_index + 1
    data['index'] = new_index
    pd.DataFrame([data]).to_sql('football_data', con=engine, if_exists='append', index=False)
    return redirect(url_for('display_data'))




@app.route('/delete_data/<int:row_index>', methods=['POST'])
def delete_data(row_index):
    print("pain")
    print(row_index)
    try:
        with engine.connect() as connection:
            delete_query = text(f"DELETE FROM football_data WHERE `index` = {row_index};")
            

            print(delete_query)
            result = connection.execute(delete_query)
            if result.rowcount > 0:
                print(f"Deleted {result.rowcount} row(s) successfully.")
            else:
                print("No rows were deleted.")
            connection.commit() 
        return redirect(url_for('display_data'))
    except exc.SQLAlchemyError as e:
        # Handle any potential errors (e.g., database connection issues)
        print(f"Error deleting data: {e}")
        return render_template('error.html', error_message='Error deleting data')


def fetch_and_insert_api_data(selected_number,selected_team):
    url = 'https://api.football-data.org/v4/teams/' + selected_team + '/matches'
    params = {'status': 'FINISHED', 'limit': selected_number} 
    headers = {'X-Auth-Token': '5981eacaedd4458f9415848e00f97473'}
    response = requests.get(url, params=params, headers=headers)

    if response.status_code == 200:
        data = response.json()
        api_data = data
        matches = api_data['matches']
        for match in matches:
            date_time = match['utcDate']
            date, time = date_time.split('T')
            home_team = match['homeTeam']['name']
            away_team = match['awayTeam']['name']
            fthg = match['score']['fullTime']['home']
            ftag = match['score']['fullTime']['away']
            duplicate_check_query = f"SELECT * FROM football_data WHERE Date = '{pd.to_datetime(date).strftime('%d/%m/%Y')}' AND HomeTeam = '{home_team}' AND AwayTeam = '{away_team}'"
            existing_data = pd.read_sql_query(duplicate_check_query, engine)

            duplicate_check_query2 = f"SELECT * FROM football_data WHERE Date = '{pd.to_datetime(date).strftime('%d/%m/%Y')}' AND HomeTeam = '{away_team}' AND AwayTeam = '{home_team}'"
            existing_data2 = pd.read_sql_query(duplicate_check_query2, engine)


            if existing_data.empty and existing_data2.empty:
                df_subset = pd.DataFrame({
                    'Date': [pd.to_datetime(date).strftime('%d/%m/%Y')],
                    'Time': [pd.to_datetime(time).strftime('%H:%M')],
                    'HomeTeam': [home_team],
                    'AwayTeam': [away_team],
                    'FTHG': [fthg],
                    'FTAG': [ftag]
                })

                current_max_index = pd.read_sql_query("SELECT MAX(`index`) FROM football_data", con=engine).iloc[0, 0]
                df_subset.index = range(current_max_index + 1, current_max_index + 2)
                df_subset.to_sql('football_data', con=engine, if_exists='append', index=True, index_label='index')
                print("Data successfully written to MySQL database.")
            else:
                print("Not working")
    else:
        print(f"Error: {response.status_code}, {response.text}")

@app.route('/fetch_and_insert_api_data', methods=['GET'])
def fetch_and_insert_api_route():
    selected_number = request.args.get('number')
    team_id = request.args.get('team_id')
    fetch_and_insert_api_data(selected_number, team_id)
    return redirect(url_for('display_data'))

@app.route('/predict_result', methods=['POST'])
def predict_result():
    hshots = int(request.form['hshots'])
    ashots = int(request.form['ashots'])
    hstarget = int(request.form['hstarget'])
    astarget = int(request.form['astarget'])
    hy = int(request.form['hy'])
    ay = int(request.form['ay'])
    hr = int(request.form['hr'])
    ar = int(request.form['ar'])

    input_data = [[hshots, ashots, hstarget, astarget, hy, ay, hr, ar]]
    prediction = model.predict(input_data)

    input_data = [hshots, ashots, hstarget, astarget, hy, ay, hr, ar]
    if prediction == 1:
        result="Home Team Wins"
    elif prediction == 2:
        result = "Away Team Wins"
    else:
        result = "Draw"
    print(input_data)

    custom_game = {
    'hshots': hshots,
    'ashots': ashots,
    'hstarget': hstarget,
    'astarget': astarget,
    'hy': hy,
    'ay': ay,
    'hr': hr,
    'ar': ar
    }

    custom_game_df = pd.DataFrame([custom_game])


    custom_game_json = jsonify(custom_game).data.decode('utf-8')
    return redirect(url_for('display_data', prediction=result, input_data=",".join([str(i) for i in input_data]), custom_game_json=custom_game_json))


if __name__ == '__main__':
    app.run('0.0.0.0', debug=True)