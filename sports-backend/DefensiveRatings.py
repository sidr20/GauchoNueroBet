from nba_api.stats.endpoints import teamestimatedmetrics
import pandas as pd

# Define the range of seasons
start_year = 2003
end_year = 2025
seasons = [f'{year}-{str(year+1)[-2:]}' for year in range(start_year, end_year)]

# Initialize an empty DataFrame to store the results
all_defensive_ratings = pd.DataFrame()

# Loop through each season and fetch the estimated defensive ratings
for season in seasons:
    team_metrics = teamestimatedmetrics.TeamEstimatedMetrics(season=season)
    df = team_metrics.get_data_frames()[0]
    df['SEASON'] = season
    all_defensive_ratings = pd.concat([all_defensive_ratings, df[['TEAM_NAME', 'E_DEF_RATING', 'SEASON']]])

# Reset the index of the final DataFrame
all_defensive_ratings.reset_index(drop=True, inplace=True)
print(all_defensive_ratings.head())
# Save the results to a CSV file
# all_defensive_ratings.to_csv('estimated_defensive_stats_since_2003.csv', index=False)
