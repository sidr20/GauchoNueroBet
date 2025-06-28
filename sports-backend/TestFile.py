from nba_api.stats.static import players
from nba_api.stats.endpoints import leaguegamefinder
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.regularizers import l2
from tensorflow.keras import Input
import random
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import shapiro

all_players = players.get_players()
player_dict = {player['full_name'].upper(): player['id'] for player in all_players}

columns = ["SEASON_ID", "TEAM_ID", "TEAM_ABBREVIATION", "TEAM_NAME", "GAME_ID", "GAME_DATE", "MATCHUP", "WL", "MIN", "PTS", "FGM", "FGA", "FG_PCT", 
    "FG3M", "FG3A", "FG3_PCT", "FTM", "FTA", "FT_PCT", "OREB", "DREB", "REB", "AST", "STL", "BLK", "TO", "PF", "PLUS_MINUS"]

features = ["Rolling_PTS", "Rolling_AST", "Rolling_REB", "MIN", "FGM", "FGA", "FG_PCT", "FG3_PCT", "FT_PCT", "FTM", "TO", "FG3M", 
            "PLUS_MINUS", "HOME_GAME", "Back_to_Back", "E_DEF_RATING"]
target = "PTS"
all_defensive_ratings = pd.read_csv("estimated_defensive_ratings_since_2003.csv")
all_defensive_ratings.rename(columns={"SEASON": "SEASON_ID"}, inplace=True)



# Modify to take 5 player inputs - currently a 1
player_inputs = []
for i in range(5):
    player_name = input(f"Type NBA Player {i + 1}: ")
    player_inputs.append(player_name)

total_results = []

# Define the model outside the loop
def create_model():
    model = Sequential([
        Input(shape=(len(features),)),
        Dense(128, kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        LeakyReLU(),
        Dropout(0.3),
        Dense(64, kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        LeakyReLU(),
        Dropout(0.3),
        Dense(32, kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        LeakyReLU(),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mae', metrics=['mae'])
    return model

# Create the model once
model = create_model()
initial_weights = model.get_weights()  # Save the initial weights

# Loop through players
for player_To_Check in player_inputs:
    player_id = player_dict.get(player_To_Check.upper())

    if not player_id:
        print(f"Player {player_To_Check} not found. Skipping...")
        continue

    # Fetch game data for the player
    gamefinder = leaguegamefinder.LeagueGameFinder(player_id_nullable=player_id)
    games = gamefinder.get_dict()['resultSets'][0]['rowSet']
    player_data = pd.DataFrame(games, columns=columns)

    # Preprocess data
    player_data["GAME_DATE"] = pd.to_datetime(player_data["GAME_DATE"])
    player_data = player_data.sort_values(by="GAME_DATE")

    player_data["Rolling_PTS"] = player_data["PTS"].rolling(window=3, min_periods=1).mean()
    player_data["Rolling_AST"] = player_data["AST"].rolling(window=3, min_periods=1).mean()
    player_data["Rolling_REB"] = player_data["REB"].rolling(window=3, min_periods=1).mean()

    player_data['HOME_GAME'] = player_data['MATCHUP'].apply(lambda x: 1 if "vs." in x else 0)
    player_data['Back_to_Back'] = player_data["GAME_DATE"].diff().dt.days.fillna(0).apply(lambda x: 1 if x == 1 else 0)

    # Clean TEAM_NAME and SEASON_ID columns
    player_data["TEAM_NAME"] = player_data["TEAM_NAME"].str.strip().str.upper()
    all_defensive_ratings["TEAM_NAME"] = all_defensive_ratings["TEAM_NAME"].str.strip().str.upper()

    player_data["SEASON_ID"] = player_data["SEASON_ID"].astype(str)
    all_defensive_ratings["SEASON_ID"] = all_defensive_ratings["SEASON_ID"].astype(str)

    # Merge defensive ratings
    player_data = player_data.merge(all_defensive_ratings, on=["TEAM_NAME", "SEASON_ID"], how="left")
    player_data = player_data.dropna(subset=["PTS"])

    # Determine the current season
    player_data["SEASON_YEAR"] = player_data["SEASON_ID"].str[-4:].astype(int)
    current_season = player_data["SEASON_YEAR"].max()
    current_season_data = player_data[player_data["SEASON_YEAR"] == current_season]

    # Exclude dates where 2 or more games were played
    game_counts = current_season_data["GAME_DATE"].value_counts()
    valid_dates = game_counts[game_counts == 1].index
    current_season_data = current_season_data[current_season_data["GAME_DATE"].isin(valid_dates)]

    # Randomly select up to 5 unique dates - currently a 1
    random_dates = random.sample(list(current_season_data["GAME_DATE"].unique()), min(5, len(current_season_data["GAME_DATE"].unique())))
    

    for target_date in random_dates:
        print(f"\nChecking player: {player_To_Check}")
        print(f"\nProcessing target date: {target_date.date()}")  # Debug print

        # Filter training and testing data
        train_data = player_data[player_data["GAME_DATE"] < target_date]
        test_data = player_data[player_data["GAME_DATE"] == target_date]

        # Debug prints for train_data
        #print(f"Number of games in train_data: {len(train_data)}")  # Size of train_data
        #if not train_data.empty:
            #print(f"Train data range: {train_data['GAME_DATE'].min().date()} to {train_data['GAME_DATE'].max().date()}")  # Date range

        # Debug prints for test_data
        #print(f"Number of games in test_data: {len(test_data)}")  # Size of test_data
        #if not test_data.empty:
            #print(f"Test data date: {test_data['GAME_DATE'].iloc[0].date()}")  # Target date

        #if train_data.empty or test_data.empty:
            #print("Skipping due to insufficient data.")
            #continue

        train_data = train_data.copy()
        test_data = test_data.copy()
        train_data.loc[:, features] = train_data[features].fillna(0)
        test_data.loc[:, features] = test_data[features].fillna(0)

        X_train = train_data[features]
        y_train = train_data[target]

        X_test = test_data[features]
        y_test = test_data[target]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Reset the model's weights before training
        model.set_weights(initial_weights)

        # Train the model
        model.fit(X_train_scaled, y_train, epochs=50, batch_size=16, verbose=0)

        y_pred = model.predict(X_test_scaled)

        actual_points = y_test.values[0]
        predicted_points = y_pred[0][0]

        total_results.append({
            "Player": player_To_Check,
            "Date": target_date.date(),
            "Actual Points": actual_points,
            "Predicted Points": predicted_points,
            "Prediction Error": abs(actual_points - predicted_points)
        })


# Calculate total errors across all players
total_rmse = np.sqrt(mean_squared_error([result['Actual Points'] for result in total_results],
                                        [result['Predicted Points'] for result in total_results]))
total_mbe = np.mean([result['Predicted Points'] - result['Actual Points'] for result in total_results])
total_r2 = r2_score([result['Actual Points'] for result in total_results],
                    [result['Predicted Points'] for result in total_results])

print("\n--- Total Results Across All Players ---")
print(f"Total RMSE: {total_rmse:.2f}")
print(f"Total MBE: {total_mbe:.2f}")
print(f"Total R^2 Score: {total_r2:.2f}")