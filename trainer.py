import xgboost as xgb
import numpy as np
import time
import pandas as pd
from scraper import Scraper
from nba_api.stats.static import teams
from nba_api.stats.endpoints import leaguegamefinder, scheduleleaguev2

def createGamesPool():
    # Loading every game from the 2023-24, 2024-25, and 2025-26 seasons
    # into DataFrames
    seasons = ["2023-24", "2024-25", "2025-26"]
    games = []

    # 6 API calls + 2 one second sleep commands
    for season in seasons:
        another_temp = scheduleleaguev2.ScheduleLeagueV2(season=season)
        time.sleep(1)
        season_start = pd.to_datetime(another_temp.get_data_frames()[1]["startDate"].iloc[0])
        temp = leaguegamefinder.LeagueGameFinder(
            league_id_nullable="00",
            season_type_nullable="Regular Season",
            season_nullable=season,
            date_from_nullable=(season_start+pd.Timedelta(weeks=3)).strftime("%m/%d/%Y")
        )
        games.append(temp.get_data_frames()[0])
        time.sleep(1)
    games_df = pd.concat(games, ignore_index=True)
    games_df["GAME_DATE"] = pd.to_datetime(games_df["GAME_DATE"])

    # games_df now has all 3 seasons worth of games
    return games_df

def createOneTrainingSet(games_df, do_not_include):
    # Create the DataFrame of games

    # Select a random game from games_df that is not specificed to ignore
    game = games_df.sample(n=1, axis="index")
    while (game["GAME_ID"].iloc[0] in do_not_include):
        game = games_df.sample(n=1, axis="index")

    # team1 is always home, team2 is always away, label is team1_score - team2_score
    if "@" in game["MATCHUP"].iloc[0]:
        team1_abbreviation = game["MATCHUP"].iloc[0][-3:]
        team2_abbreviation = game["MATCHUP"].iloc[0][:3]
        label = -game["PLUS_MINUS"].iloc[0]
    else:
        team1_abbreviation = game["MATCHUP"].iloc[0][:3]
        team2_abbreviation = game["MATCHUP"].iloc[0][-3:]
        label = game["PLUS_MINUS"].iloc[0]

    team1 = teams.find_team_by_abbreviation(team1_abbreviation)
    team2 = teams.find_team_by_abbreviation(team2_abbreviation)

    team1_scraper = Scraper(team1["full_name"])
    team2_scraper = Scraper(str(team2["full_name"]))

    # Set up calendar boundaries as strings because scraper takes in strings, not datetime objects
    end_date = game["GAME_DATE"] - pd.Timedelta(days=1)
    start_date = end_date - pd.Timedelta(weeks=3)
    start_date_str = start_date.dt.strftime("%Y-%m-%d")
    end_date_str = end_date.dt.strftime("%Y-%m-%d")

    # Find rolling stats for each team over the duration specified before
    team1_df = team1_scraper.averageStats(
        team1_scraper.gamesDateRange(start_date_str, end_date_str)
        )
    team2_df = team2_scraper.averageStats(
        team2_scraper.gamesDateRange(start_date_str, end_date_str)
        )
    
    # Label each column in team1_df and team2_df so once they merge it's clear
    # which column belongs to which team
    team1_df = team1_df.add_prefix("team1_")
    team2_df = team2_df.add_prefix("team2_")

    # Merge team1_df and team2_df and insert the actual label data
    training_df = team1_df.join(team2_df)
    training_df.insert(loc=len(training_df.columns), 
                        column="label", 
                        value=label
                        )

    # training_df now has the data needed to train the model
    return training_df, game["GAME_ID"].iloc[0]

def createNTrainingSets(N):
    # Iterate over N unique games and add their data to training_sets
    games_df = createGamesPool()
    training_sets = []
    visited = []
    for _ in range(N):
        training_df, game_id = createOneTrainingSet(games_df, visited)
        training_sets.append(training_df)
        visited.append(game_id)

    # training_sets now has all the required training data and labels
    return training_sets

print(createNTrainingSets(5))