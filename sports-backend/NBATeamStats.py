import pandas as pd
import time
from nba_api.stats.endpoints import teamyearbyyearstats

# List of NBA teams
nba_teams = [
    {"id": 1610612737, "full_name": "Atlanta Hawks"},
    {"id": 1610612738, "full_name": "Boston Celtics"},
    {"id": 1610612751, "full_name": "Brooklyn Nets"},
    {"id": 1610612766, "full_name": "Charlotte Hornets"},
    {"id": 1610612741, "full_name": "Chicago Bulls"},
    {"id": 1610612739, "full_name": "Cleveland Cavaliers"},
    {"id": 1610612742, "full_name": "Dallas Mavericks"},
    {"id": 1610612743, "full_name": "Denver Nuggets"},
    {"id": 1610612765, "full_name": "Detroit Pistons"},
    {"id": 1610612744, "full_name": "Golden State Warriors"},
    {"id": 1610612745, "full_name": "Houston Rockets"},
    {"id": 1610612754, "full_name": "Indiana Pacers"},
    {"id": 1610612746, "full_name": "LA Clippers"},
    {"id": 1610612747, "full_name": "Los Angeles Lakers"},
    {"id": 1610612763, "full_name": "Memphis Grizzlies"},
    {"id": 1610612748, "full_name": "Miami Heat"},
    {"id": 1610612749, "full_name": "Milwaukee Bucks"},
    {"id": 1610612750, "full_name": "Minnesota Timberwolves"},
    {"id": 1610612740, "full_name": "New Orleans Pelicans"},
    {"id": 1610612752, "full_name": "New York Knicks"},
    {"id": 1610612760, "full_name": "Oklahoma City Thunder"},
    {"id": 1610612753, "full_name": "Orlando Magic"},
    {"id": 1610612755, "full_name": "Philadelphia 76ers"},
    {"id": 1610612756, "full_name": "Phoenix Suns"},
    {"id": 1610612757, "full_name": "Portland Trail Blazers"},
    {"id": 1610612758, "full_name": "Sacramento Kings"},
    {"id": 1610612759, "full_name": "San Antonio Spurs"},
    {"id": 1610612761, "full_name": "Toronto Raptors"},
    {"id": 1610612762, "full_name": "Utah Jazz"},
    {"id": 1610612764, "full_name": "Washington Wizards"}
]

# Initialize an empty DataFrame to store the results
columns = ['YEAR', 'TEAM_ID', 'TEAM_NAME', 'STL', 'BLK', "WIN_PCT"]
all_team_stats = pd.DataFrame(columns=columns)

# Process teams in batches
batch_size = 15
for i in range(0, len(nba_teams), batch_size):
    batch = nba_teams[i:i + batch_size]
    for team in batch:
        team_id = team['id']
        team_name = team['full_name']
        try:
            team_stats = teamyearbyyearstats.TeamYearByYearStats(team_id=team_id).get_data_frames()[0]
            team_stats = team_stats[team_stats['YEAR'] >= '2003-04']
            team_stats = team_stats[['YEAR', 'STL', 'BLK', "WIN_PCT"]]
            team_stats['TEAM_ID'] = team_id
            team_stats['TEAM_NAME'] = team_name
            all_team_stats = pd.concat([all_team_stats, team_stats], ignore_index=True)
        except Exception as e:
            print(f"Error fetching data for {team_name}: {e}")
        time.sleep(1)  # Add a delay between requests

# Sort the DataFrame by YEAR and TEAM_NAME
all_team_stats = all_team_stats.sort_values(by=['YEAR', 'TEAM_NAME']).reset_index(drop=True)

# Save the final DataFrame to a CSV file
all_team_stats.to_csv('nba_team_stats_since_2003.csv', index=False)

# Print the first few rows of the DataFrame
print(all_team_stats.head())