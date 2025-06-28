import requests
import pandas as pd
import json
import os
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# API endpoints
odds_url = "https://api.sportsgameodds.com/v1/odds"
stats_url = "https://api.sportsgameodds.com/v2/events"

# API key
api_key = "fb47a2bb8b3b37383ae487c0f72b4dd1"

# List of players
star_players = [
    "MIKAL_BRIDGES",
    "STEPHEN_CURRY",
    "LUKA_DONCIC",
    "JAYSON_TATUM",
    "ZACH_LAVINE",
    "JALEN_GREEN",
    "DE'AARON_FOX",
    "KEVIN_DURANT",
    "DONOVAN_MITCHELL",
    "MAX_STRUS",
    "ANTHONY_EDWARDS"
]

# Headers
headers = {
    "X-API-Key": api_key,
    "Content-Type": "application/json"
}

# File paths
json_file = "nba_combined_data.json"
csv_file = "nba_combined_data.csv"

# Function to check if cached file is valid and contains all players
def is_valid_cache(file_path, players):
    if os.path.exists(file_path):
        file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
        if datetime.now() - file_time < timedelta(days=1):  # Cache is fresh
            with open(file_path, "r") as f:
                try:
                    cached_data = json.load(f)
                    cached_players = {record["Player"] for record in cached_data}
                    if all(player.replace("_", " ") in cached_players for player in players):
                        print("✅ Using cached data from:", file_time)
                        return cached_data
                    else:
                        print("⚠️ Cached data is missing some players. Fetching new data...")
                except json.JSONDecodeError:
                    print("⚠️ Cached file is corrupted. Fetching new data...")
    return None  # Fetch new data if cache is invalid

# Load cached data if valid
data = is_valid_cache(json_file, star_players)

# If no valid cached data, fetch new data
if data is None:
    print("🆕 No valid cached data found. Fetching new data...")
    game_records = []

    for player in star_players:
        odd_id = f"points-{player}_1_NBA-game-ou-under"

        # Fetch odds data
        odds_params = {
            "leagueID": "NBA",
            "limit": 15,
            "finalized": "true",
            "oddIDs": odd_id
        }

        odds_response = requests.get(odds_url, params=odds_params, headers=headers)
        print(f"\n🔍 Requesting Odds Data for {player}... Status Code: {odds_response.status_code}")

        # Fetch player stats data (last 10 games)
        stats_params = {"player": player}
        stats_response = requests.get(stats_url, params=stats_params, headers=headers)
        print(f"🔍 Requesting Stats Data for {player}... Status Code: {stats_response.status_code}")

        if odds_response.status_code == 200 and stats_response.status_code == 200:
            odds_data = odds_response.json()
            stats_data = stats_response.json()

            if odds_data.get("success") and "data" in odds_data and len(odds_data["data"]) > 0:
                for game in odds_data["data"]:
                    date = game["status"]["startsAt"][:10]

                    for odds_key, odds_info in game["odds"].items():
                        if odds_key.startswith(f"points-{player}"):
                            # Get player stats for the same game
                            player_stats = next((g["game"]["home"] for g in stats_data["results"]
                                                 if g["eventID"] == game["eventID"]), {})

                            # Combine data
                            game_records.append({
                                "Date": date,
                                "Player": player.replace("_", " "),
                                "Event ID": game["eventID"],
                                "Actual Points": odds_info.get("score", "N/A"),
                                "Expected Points (Book)": odds_info.get("bookOverUnder", "N/A"),
                                "Expected Points (Fair)": odds_info.get("fairOverUnder", "N/A"),
                                "Fair Odds": odds_info.get("fairOdds", "N/A"),
                                "Book Odds": odds_info.get("bookOdds", "N/A"),
                                "Started": odds_info.get("started", False),
                                "Ended": odds_info.get("ended", False),
                                "Cancelled": odds_info.get("cancelled", False),

                                # Individual player stats
                                "Field Goals Made": player_stats.get("fieldGoalsMade", "N/A"),
                                "Field Goals Attempted": player_stats.get("fieldGoalsAttempted", "N/A"),
                                "Three Pointers Made": player_stats.get("threePointersMade", "N/A"),
                                "Three Pointers Attempted": player_stats.get("threePointersAttempted", "N/A"),
                                "Free Throws Made": player_stats.get("freeThrowsMade", "N/A"),
                                "Free Throws Attempted": player_stats.get("freeThrowsAttempted", "N/A"),
                                "Rebounds": player_stats.get("rebounds", "N/A"),
                                "Assists": player_stats.get("assists", "N/A"),
                                "Turnovers": player_stats.get("turnovers", "N/A"),
                                "Steals": player_stats.get("steals", "N/A")
                            })

            else:
                print(f"⚠️ No valid game data found for {player} in API response.")

        else:
            print(f"❌ Failed to fetch data for {player}. Odds Status Code: {odds_response.status_code}, Stats Status Code: {stats_response.status_code}")

    # Save new data to JSON file only if we fetched records
    if game_records:
        with open(json_file, "w") as f:
            json.dump(game_records, f, indent=4)
        print("💾 New data saved to", json_file)
    else:
        print("⚠️ No valid data fetched. Keeping old cache (if any).")

    data = game_records  # Use newly fetched data

# Convert to DataFrame
df = pd.DataFrame(data)

# Ensure expected columns exist before converting to numeric
numeric_cols = [
    "bookOverUnder",  # No change
    "fairOverUnder",  # No change
    "fairOdds",  # No change
    "bookOdds",  # No change
    "fieldGoalsMade",
    "fieldGoalsAttempted",
    "threePointersMade",
    "threePointersAttempted",
    "freeThrowsMade",
    "freeThrowsAttempted",
    "rebounds",
    "assists",
    "turnovers",
    "steals"
]


for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").round(2)

# Display DataFrame in Jupyter Notebook (if available)
try:
    from IPython.display import display
    display(df)
except ImportError:
    print(df)

# Save DataFrame as CSV
df.to_csv(csv_file, index=False)
print("📊 Data saved to", csv_file)

df = df[df["Actual Points"] != 0]
df["Actual Points"] = pd.to_numeric(df["Actual Points"], errors = "coerce")
df["Expected Points (Book)"] = pd.to_numeric(df["Expected Points (Book)"], errors="coerce")
df["Expected Points (Fair)"] = pd.to_numeric(df["Expected Points (Fair)"], errors="coerce")
df["Fair Odds"] = pd.to_numeric(df["Fair Odds"], errors="coerce")
df["Book Odds"] = pd.to_numeric(df["Book Odds"], errors="coerce")

df["Rolling Avg Points"] = df.groupby("Player")["Actual Points"].transform(lambda x: x.rolling(5, min_periods=1).mean())
df["Rolling Avg Expected Points"] = df.groupby("Player")["Expected Points (Book)"].transform(lambda x: x.rolling(5, min_periods=1).mean())

features = [
    "Expected Points (Book)",
    "Expected Points (Fair)",
    "Fair Odds",
    "Book Odds",
    "Rolling Avg Points",
    "Rolling Avg Expected Points"
]
target = "Actual Points"

df = df.dropna(subset=features + [target])

X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Accuracy: " + mse)
print("R^2:" + r2)