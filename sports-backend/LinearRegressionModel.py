from nba_api.stats.static import players
from nba_api.stats.endpoints import leaguegamefinder, playergamelog
from nba_api.stats.library.parameters import SeasonAll
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import math



# Fetch all players
all_players = players.get_players()
player_dict = {player['full_name'].upper(): player['id'] for player in all_players}

# Get player input
player_To_Check = input("Type an NBA Player:")
player_id = player_dict.get(player_To_Check.upper())

if not player_id:
    print("Player not found.")
    exit()

# Fetch game logs
gamefinder = leaguegamefinder.LeagueGameFinder(player_id_nullable=player_id)
games = gamefinder.get_dict()['resultSets'][0]['rowSet']

columns = ["SEASON_ID", "TEAM_ID", "TEAM_ABREVIATION", "TEAM_NAME", "GAME_ID", "GAME_DATE", "MATCHUP", "WL", "MIN", "PTS", "FGM", "FGA", "FG_PCT", 
    "FG3M", "FG3A", "FG3_PCT", "FTM", "FTA", "FT_PCT", "OREB", "DREB", 
    "REB", "AST", "STL", "BLK", "TO", "PF", "PLUS_MINUS"]
data = pd.DataFrame(games, columns=columns)

# Convert game date and sort
data["GAME_DATE"] = pd.to_datetime(data["GAME_DATE"])
data = data.sort_values(by="GAME_DATE")

# Rolling stats (short-term and long-term trends)
data["Rolling_PTS_5"] = data["PTS"].rolling(window=5, min_periods=1).mean()
data["Rolling_AST_3"] = data["AST"].rolling(window=3, min_periods=1).mean()
data["Rolling_REB_3"] = data["REB"].rolling(window=3, min_periods=1).mean()

# Lag features (previous game stats)
data["PTS_Lag1"] = data["PTS"].shift(1)

data["MIN_Lag1"] = data["MIN"].shift(1)
data["FG_PCT_Lag1"] = data["FG_PCT"].shift(1)

# Exponential Moving Average (EMA) for recent trends
data["EMA_PTS_5"] = data["PTS"].ewm(span=5, adjust=False).mean()

# Fill missing lag values (first game might have NaN for lag)
data = data.fillna(method='bfill')

# Drop rows with NaNs in target variable
data = data.dropna(subset=["PTS"])

# Feature selection
features = ["Rolling_PTS_5", "Rolling_AST_3", "Rolling_REB_3", "MIN", "FG_PCT", "FG3_PCT", "FT_PCT", "PLUS_MINUS", 
            "PTS_Lag1", "MIN_Lag1", "FG_PCT_Lag1", "EMA_PTS_5"]
target = "PTS"

data[features] = data[features].fillna(0)

# Prepare data for training
X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Model Accurately Predicted Within {math.sqrt(mse):.2f} Points\nR²: {r2:.2f}")

# Predict next game performance
latest_game_features = data[features].iloc[-1].values.reshape(1, -1)
predicted_points = model.predict(latest_game_features)

scoring_range_negative = predicted_points[0] - math.sqrt(mse)
scoring_range_positive = predicted_points[0] + math.sqrt(mse)

print(f"Predicted Points for {player_To_Check} in the next game: {predicted_points[0]:.2f}")
print(f'{player_To_Check.upper()} should score within {scoring_range_negative:.2f} - {scoring_range_positive:.2f} points')
