import xgboost as xgb
import pandas as pd
from scraper import Scraper
import numpy as np
from datetime import datetime
from sklearn.metrics import mean_squared_error
from scipy.stats import norm
from analysis import plotWinOdds

# Get user input for home team, away team, and game date
home_team = input("Home Team: ")
away_team = input("Away Team: ")
game_date = pd.to_datetime(input("Game Date (YYYY-MM-DD): "), format="%Y-%m-%d")

# Import model
model = xgb.XGBRegressor()
model.load_model(fname="model.json")
if (game_date.month >= 9):
    start_date = f"{game_date.year}-09-01"
else:
    start_date = f"{game_date.year - 1}-09-01"
end_date = datetime.strftime(game_date - pd.Timedelta(days=1), "%Y-%m-%d")

# Get team data and construct row to be the same as training format
home_scraper = Scraper(home_team)
away_scraper = Scraper(away_team)
home_games = home_scraper.gamesDateRange(start_date, end_date)
away_games = away_scraper.gamesDateRange(start_date, end_date)
home_df = home_scraper.averageStats(home_games).add_prefix("team1_")
away_df = away_scraper.averageStats(away_games).add_prefix("team2_")
row = pd.concat([home_df, away_df], axis=1)
row["REST_DIFF"] = (home_scraper.restDays(game_date, home_games)
                    - away_scraper.restDays(game_date, away_games))

# Get +/- and win probability from the model
predictions = model.predict(row)
margin = predictions[0]
results = pd.read_csv("predictions.csv")
rmse = np.sqrt(mean_squared_error(results["actual"], results["prediction"]))
win_prob = norm.cdf(margin / rmse)

# Print result and terminate
if margin >= 0:
    print(f"{home_team} by {margin:.1f}, {win_prob:.0%} to win")
else:
    print(f"{away_team} by {-margin:.1f}, {1 - win_prob:.0%} to win")

plotWinOdds(margin, rmse, home_team, away_team)