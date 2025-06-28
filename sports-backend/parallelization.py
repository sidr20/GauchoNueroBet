from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import leaguegamefinder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import shapiro
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.regularizers import l2
from tensorflow.keras import Input
import random
from concurrent.futures import ThreadPoolExecutor
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

# Retrieve players and player id
all_players = players.get_players()
player_dict = {player['full_name'].upper(): player['id'] for player in all_players}

player_To_Check = input("Type an NBA Player: ")
player_id = player_dict.get(player_To_Check.upper())

if not player_id:
    raise ValueError("Player not found. Please check the spelling.")

# Build mapping from team abbreviation to full team name
all_teams = teams.get_teams()
team_abbrev_to_name = {team["abbreviation"]: team["full_name"] for team in all_teams}

# Get games for the player
gamefinder = leaguegamefinder.LeagueGameFinder(player_id_nullable=player_id)
games = gamefinder.get_dict()['resultSets'][0]['rowSet']
columns = ["SEASON_ID", "TEAM_ID", "TEAM_ABBREVIATION", "TEAM_NAME", "GAME_ID", "GAME_DATE",
           "MATCHUP", "WL", "MIN", "PTS", "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT",
           "FTM", "FTA", "FT_PCT", "OREB", "DREB", "REB", "AST", "STL", "BLK", "TO", "PF", "PLUS_MINUS"]
data = pd.DataFrame(games, columns=columns)

data["GAME_DATE"] = pd.to_datetime(data["GAME_DATE"])
data = data.sort_values(by="GAME_DATE")

# Create rolling averages for points, assists, and rebounds
data["Rolling_PTS"] = data["PTS"].rolling(window=3, min_periods=1).mean()
data["Rolling_AST"] = data["AST"].rolling(window=3, min_periods=1).mean()
data["Rolling_REB"] = data["REB"].rolling(window=3, min_periods=1).mean()

# Identify home games and back-to-back scenarios
data['HOME_GAME'] = data['MATCHUP'].apply(lambda x: 1 if "vs." in x else 0)
data['Back_to_Back'] = data['GAME_DATE'].diff().dt.days.fillna(0).apply(lambda x: 1 if x == 1 else 0)

# Extract opponent team from the MATCHUP column and convert to full team name
def extract_opponent(matchup):
    if "vs." in matchup:
        opp_abbrev = matchup.split("vs. ")[1]
    elif "@" in matchup:
        opp_abbrev = matchup.split("@ ")[1]
    else:
        opp_abbrev = None
    # Use the mapping to get the full team name; if not found, return the abbreviation
    return team_abbrev_to_name.get(opp_abbrev, opp_abbrev)

data['OPPONENT'] = data['MATCHUP'].apply(extract_opponent)

def convert_season_id(season_id):
    year = int(season_id[-4:])  # Extract the last 4 digits
    return f"{year}-{str(year + 1)[-2:]}"  # Format as "xxxx-xx"

data["SEASON_ID"] = data["SEASON_ID"].apply(convert_season_id)

# Load opponent defensive ratings and rename season column to match our data
all_defensive_ratings = pd.read_csv("estimated_defensive_ratings_since_2003.csv")

all_defensive_ratings.rename(columns={"SEASON": "SEASON_ID"}, inplace=True)
# Keep only the necessary columns from defensive ratings
opp_def_stats = all_defensive_ratings[['TEAM_NAME', 'SEASON_ID', 'E_DEF_RATING']]

team_name_replacements = {
    "Charlotte Bobcats": "Charlotte Hornets",
    "Seattle SuperSonics": "Oklahoma City Thunder",
    "New Orleans/Oklahoma City Hornets": "New Orleans Pelicans",
    "New Jersey Nets": "Brooklyn Nets",
    "New Orleans Hornets": "New Orleans Pelicans"
}

# Replace team names in the TEAM_NAME column
opp_def_stats["TEAM_NAME"] = opp_def_stats["TEAM_NAME"].replace(team_name_replacements)

# Merge opponent defensive rating stats on opponent full name and season
data = data.merge(opp_def_stats,left_on=["OPPONENT", "SEASON_ID"],right_on=["TEAM_NAME", "SEASON_ID"],
    how="left",
    suffixes=("", "_opp")
)

# Rename the defensive rating column
data.rename(columns={"E_DEF_RATING": "OPP_E_DEF_RATING"}, inplace=True)
data.drop(columns=["TEAM_NAME_opp"], inplace=True)

valid_team_names = all_defensive_ratings["TEAM_NAME"].unique()
data = data[data["OPPONENT"].isin(valid_team_names)]

# Load opponent team stats and adjust column names
all_team_stats = pd.read_csv("nba_team_stats_since_2003.csv")
all_team_stats.rename(columns={"YEAR": "SEASON_ID"}, inplace=True)
all_team_stats.rename(columns={"STL": "TEAM_STL", "BLK": "TEAM_BLK", "WIN_PCT": "TEAM_WIN_PCT"}, inplace=True)
# Keep only needed columns for the opponent
opp_team_stats = all_team_stats[['TEAM_NAME', 'SEASON_ID', 'TEAM_STL', 'TEAM_BLK', 'TEAM_WIN_PCT']]


data["OPPONENT"] = data["OPPONENT"].replace({"Los Angeles Clippers": "LA Clippers"})
# Merge opponent team stats on opponent full name and season
data = data.merge(opp_team_stats, left_on=["OPPONENT", "SEASON_ID"], right_on=["TEAM_NAME", "SEASON_ID"],
                  how="left", suffixes=("", "_opp"))
data.rename(columns={"TEAM_STL": "OPP_TEAM_STL", "TEAM_BLK": "OPP_TEAM_BLK", "TEAM_WIN_PCT": "OPP_TEAM_WIN_PCT"}, inplace=True)
data.drop(columns=["TEAM_NAME_opp"], inplace=True)

# Fill missing values for opponent stats with their mean values
data["OPP_E_DEF_RATING"] = data["OPP_E_DEF_RATING"].fillna(data["OPP_E_DEF_RATING"].mean())
data["OPP_TEAM_STL"] = data["OPP_TEAM_STL"].fillna(data["OPP_TEAM_STL"].mean())
data["OPP_TEAM_BLK"] = data["OPP_TEAM_BLK"].fillna(data["OPP_TEAM_BLK"].mean())
data["OPP_TEAM_WIN_PCT"] = data["OPP_TEAM_WIN_PCT"].fillna(data["OPP_TEAM_WIN_PCT"].mean())

# Drop any remaining rows without points (if any)
data = data.dropna(subset=["PTS"])
# Update the features list to include opponent stats instead of the player's own defensive metrics
features = ["Rolling_PTS", "Rolling_AST", "Rolling_REB", "MIN", "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT",
           "FTM", "FTA", "FT_PCT", "OREB", "DREB", "STL", "BLK", "TO", "PF", "PLUS_MINUS", "HOME_GAME", "Back_to_Back",
            "OPP_E_DEF_RATING", "OPP_TEAM_STL", "OPP_TEAM_BLK", "OPP_TEAM_WIN_PCT"]
target = "PTS"

data[features] = data[features].fillna(0)

# Prepare training data
X = data[features]
y = data[target]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Build the neural network model
model = Sequential([
    Dense(128, input_shape=(X_train.shape[1],), kernel_regularizer=l2(0.01)),
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

model.fit(X_train, y_train, epochs=100, batch_size=16, verbose=1, validation_data=(X_test, y_test))

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

# For prediction, we take the average of the last 10 games
next_game_features = data[features].iloc[-10:].mean()
next_game_features_scaled = scaler.transform([next_game_features])
predicted_points = model.predict(next_game_features_scaled)[0][0]

print(f"Predicted Points for {player_To_Check.upper()} in the next game: {predicted_points:.2f}")
print(f"On average, the model's predictions are off by {mae:.2f} points.")
print(f'So...{player_To_Check.upper()} should score within {predicted_points - mae:.2f} - {predicted_points + mae:.2f} points')

# ----------------------------------------------THIS PART IS FOR TESTING: UN-COMMENT TO TEST-------------------------------------------------


# INPUT 5 PLAYERS OF DIFFERENT SKILL LEVELS
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

# Save the initial weights to ensure each thread starts from the same state
base_model = create_model()
initial_weights = base_model.get_weights()

# Define the function to process a single player
def process_player(player_To_Check):
    player_id = player_dict.get(player_To_Check.upper())
    if not player_id:
        print(f"Player {player_To_Check} not found. Skipping...")
        return []  # Return an empty list if player not found

    # Fetch game data for the player
    gamefinder = leaguegamefinder.LeagueGameFinder(player_id_nullable=player_id)
    games = gamefinder.get_dict()['resultSets'][0]['rowSet']
    columns = ["SEASON_ID", "TEAM_ID", "TEAM_ABBREVIATION", "TEAM_NAME", "GAME_ID", "GAME_DATE",
               "MATCHUP", "WL", "MIN", "PTS", "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT",
               "FTM", "FTA", "FT_PCT", "OREB", "DREB", "REB", "AST", "STL", "BLK", "TO", "PF", "PLUS_MINUS"]
    data = pd.DataFrame(games, columns=columns)
    
    data["GAME_DATE"] = pd.to_datetime(data["GAME_DATE"])
    data = data.sort_values(by="GAME_DATE")
    
    # Create rolling averages for points, assists, and rebounds
    data["Rolling_PTS"] = data["PTS"].rolling(window=3, min_periods=1).mean()
    data["Rolling_AST"] = data["AST"].rolling(window=3, min_periods=1).mean()
    data["Rolling_REB"] = data["REB"].rolling(window=3, min_periods=1).mean()
    
    # Identify home games and back-to-back scenarios
    data['HOME_GAME'] = data['MATCHUP'].apply(lambda x: 1 if "vs." in x else 0)
    data['Back_to_Back'] = data["GAME_DATE"].diff().dt.days.fillna(0).apply(lambda x: 1 if x == 1 else 0)
    
    # Extract opponent team and convert season format
    data['OPPONENT'] = data['MATCHUP'].apply(extract_opponent)
    data["SEASON_ID"] = data["SEASON_ID"].apply(convert_season_id)
    
    # Merge opponent defensive ratings
    data = data.merge(opp_def_stats, left_on=["OPPONENT", "SEASON_ID"], 
                      right_on=["TEAM_NAME", "SEASON_ID"],
                      how="left", suffixes=("", "_opp"))
    data.rename(columns={"E_DEF_RATING": "OPP_E_DEF_RATING"}, inplace=True)
    data.drop(columns=["TEAM_NAME_opp"], inplace=True)
    
    # Merge opponent team stats
    data = data.merge(opp_team_stats, left_on=["OPPONENT", "SEASON_ID"],
                      right_on=["TEAM_NAME", "SEASON_ID"],
                      how="left", suffixes=("", "_opp"))
    data.rename(columns={"TEAM_STL": "OPP_TEAM_STL", 
                         "TEAM_BLK": "OPP_TEAM_BLK", 
                         "TEAM_WIN_PCT": "OPP_TEAM_WIN_PCT"}, inplace=True)
    data.drop(columns=["TEAM_NAME_opp"], inplace=True)
    
    # Fill missing values for opponent stats with mean values
    data["OPP_E_DEF_RATING"] = data["OPP_E_DEF_RATING"].fillna(data["OPP_E_DEF_RATING"].mean())
    data["OPP_TEAM_STL"] = data["OPP_TEAM_STL"].fillna(data["OPP_TEAM_STL"].mean())
    data["OPP_TEAM_BLK"] = data["OPP_TEAM_BLK"].fillna(data["OPP_TEAM_BLK"].mean())
    data["OPP_TEAM_WIN_PCT"] = data["OPP_TEAM_WIN_PCT"].fillna(data["OPP_TEAM_WIN_PCT"].mean())
    
    # Drop rows without points
    data = data.dropna(subset=["PTS"])
    
    # Determine the current season and filter valid game dates
    current_season = data["SEASON_ID"].max()
    current_season_data = data[data["SEASON_ID"] == current_season]
    game_counts = current_season_data["GAME_DATE"].value_counts()
    valid_dates = game_counts[game_counts == 1].index
    current_season_data = current_season_data[current_season_data["GAME_DATE"].isin(valid_dates)]
    
    # Randomly select up to 5 unique dates
    if len(current_season_data["GAME_DATE"].unique()) < 1:
        return []  # Not enough valid dates for testing
    random_dates = random.sample(list(current_season_data["GAME_DATE"].unique()), 
                                 min(10, len(current_season_data["GAME_DATE"].unique())))
    
    player_results = []
    for target_date in random_dates:
        train_data = data[data["GAME_DATE"] < target_date]
        test_data = data[data["GAME_DATE"] == target_date]
        if train_data.empty or test_data.empty:
            continue

        # Ensure features are filled
        train_data.loc[:, features] = train_data[features].fillna(0)
        test_data.loc[:, features] = test_data[features].fillna(0)

        X_train = train_data[features]
        y_train = train_data[target]
        X_test = test_data[features]
        y_test = test_data[target]

        # Scale data
        scaler_local = StandardScaler()
        X_train_scaled = scaler_local.fit_transform(X_train)
        X_test_scaled = scaler_local.transform(X_test)

        # Create a new model instance and set initial weights
        local_model = create_model()
        local_model.set_weights(initial_weights)
        
        # Train the model for a fixed number of epochs
        local_model.fit(X_train_scaled, y_train, epochs=50, batch_size=16, verbose=0)
        y_pred = local_model.predict(X_test_scaled)
        predicted_points = y_pred[0][0]
        actual_points = y_test.values[0]
        
        player_results.append({
            "Player": player_To_Check,
            "Date": target_date.date(),
            "Actual Points": actual_points,
            "Predicted Points": predicted_points,
            "Prediction Error": abs(actual_points - predicted_points)
        })
    return player_results

# Define player inputs
player_inputs = [player_To_Check]

# Process players in parallel using ThreadPoolExecutor
with ThreadPoolExecutor() as executor:
    all_results = list(executor.map(process_player, player_inputs))

# Flatten the list of results
total_results = [result for player_results in all_results if player_results for result in player_results]

# Calculate total errors across all players
total_mae = mean_absolute_error([result['Actual Points'] for result in total_results],
                                [result['Predicted Points'] for result in total_results])
total_mbe = np.mean([result['Predicted Points'] - result['Actual Points'] for result in total_results])
total_r2 = r2_score([result['Actual Points'] for result in total_results],
                    [result['Predicted Points'] for result in total_results])

print("\n--- Total Results Across All Players ---")
print(f"Total MAE: {total_mae:.2f}")
print(f"Total MBE: {total_mbe:.2f}")
print(f"Total R^2 Score: {total_r2:.2f}")