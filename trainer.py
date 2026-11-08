import time
import pandas as pd
from scraper import Scraper, fetchWithRetry
from nba_api.stats.static import teams
from nba_api.stats.endpoints import leaguegamefinder

'''
Build the pool of eligible historical games to sample training examples
from. Pulls every Regular Season game from the 2023-24, 2024-25, and
2025-26 seasons, excluding each season's first 3 weeks (to avoid rolling
stats windows leaking into the prior season/offseason).

@return     DataFrame of all eligible games across the 3 seasons
@author     Arnav Prasad
@version    Sep 21, 2026
'''
def createGamesPool():
    seasons = ["2023-24", "2024-25", "2025-26"]
    games = []

    # Pull every game from these seasons
    for season in seasons:
        temp = fetchWithRetry(lambda season=season: leaguegamefinder.LeagueGameFinder(
            league_id_nullable="00",
            season_type_nullable="Regular Season",
            season_nullable=season,
        ))
        season_games = temp.get_data_frames()[0]
        season_games["GAME_DATE"] = pd.to_datetime(season_games["GAME_DATE"])

        # Season start is the earliest game of the season
        season_start = season_games["GAME_DATE"].min()
        season_games["SEASON_START"] = season_start

        # Keep only games at least 3 weeks into the season for better averages
        season_games = season_games[season_games["GAME_DATE"] >= season_start + pd.Timedelta(weeks=3)]
        games.append(season_games)
        time.sleep(1)

    games_df = pd.concat(games, ignore_index=True)
    return games_df

'''
Build one training row from a randomly sampled game in games_df. team1 is
always the home team and team2 is always the away team; label is
team1_score - team2_score. Features are each team's rolling stats
averaged over the 3 weeks prior to the sampled game (to avoid leakage).
GAME_ID is included as the first column for traceability (drop it before
training/predicting with the model).

@param games_df         Pool of eligible games to sample from (from createGamesPool)
@param do_not_include    List of GAME_IDs to exclude from sampling (already-used games)
@return                  One-row training DataFrame (includes GAME_ID, REST_DIFF, and label)
@author                  Arnav Prasad
@version                 Sep 22, 2026
'''
def createOneTrainingSet(games_df, do_not_include):
    # Ignore certain games
    game = games_df.sample(n=1, axis="index")
    while (game["GAME_ID"].iloc[0] in do_not_include):
        game = games_df.sample(n=1, axis="index")

    # Identify and classify team1 and team2 based on home status
    if "@" in game["MATCHUP"].iloc[0]:
        team1_abbreviation = game["MATCHUP"].iloc[0][-3:]
        team2_abbreviation = game["MATCHUP"].iloc[0][:3]
        label = -game["PLUS_MINUS"].iloc[0]
    else:
        team1_abbreviation = game["MATCHUP"].iloc[0][:3]
        team2_abbreviation = game["MATCHUP"].iloc[0][-3:]
        label = game["PLUS_MINUS"].iloc[0]

    # Get full team name and initializing scraper objects
    team1 = teams.find_team_by_abbreviation(team1_abbreviation)
    team2 = teams.find_team_by_abbreviation(team2_abbreviation)
    team1_scraper = Scraper(team1["full_name"])
    team2_scraper = Scraper(team2["full_name"])

    # Set up dates of interest as both datetime objects and strings
    game_date = game["GAME_DATE"].iloc[0]
    end_date = game["GAME_DATE"] - pd.Timedelta(days=1)
    start_date = game["SEASON_START"]
    start_date_str = start_date.dt.strftime("%Y-%m-%d")
    end_date_str = end_date.dt.strftime("%Y-%m-%d")

    # Get team 1 and team 2 recent games
    team1_games = team1_scraper.gamesDateRange(start_date_str.iloc[0], end_date_str.iloc[0])
    team2_games = team2_scraper.gamesDateRange(start_date_str.iloc[0], end_date_str.iloc[0])

    # Get rolling stats
    team1_df = team1_scraper.averageStats(team1_games)
    team2_df = team2_scraper.averageStats(team2_games)

    # Fix labels
    team1_df = team1_df.add_prefix("team1_")
    team2_df = team2_df.add_prefix("team2_")

    training_df = team1_df.join(team2_df)

    # Calculate rest_differential and add to training_df
    rest_differential = team1_scraper.restDays(game_date, team1_games) - team2_scraper.restDays(game_date, team2_games)
    training_df.insert(loc=len(training_df.columns),
                        column="REST_DIFF",
                        value=rest_differential)

    # Add label to training_df
    training_df.insert(loc=len(training_df.columns),
                        column="label",
                        value=label
                        )
    # Add GAME_ID to training_df
    training_df.insert(loc=0,
                       column="GAME_ID",
                       value=game["GAME_ID"].iloc[0]
                       )

    # training_df now contains all information of interest
    return training_df

'''
Build N unique training rows by repeatedly sampling distinct games from a
single shared games pool (built once, not per-row), writing to disk in
checkpoints of checkpoint_size rows instead of only once at the very end.
This means a crash partway through only costs the rows since the last
checkpoint, not the entire run. Either overwrites filename at the start of the
run, then appends each subsequent checkpoint onto it, or appends each checkpoint
to filename.

@param N                 number of training rows to generate
@param checkpoint_size   how many rows to accumulate before writing to disk
@param filename          path of the CSV file to write to
@param resume            whether to append to filename or rewrite filename
@return                  DataFrame of all N training rows generated this run
@author                  Arnav Prasad
@version                 Sep 22, 2026
'''
def createNTrainingSets(N, checkpoint_size=50, filename="training_data.csv", resume=False):
    # Get every game, create a list of games visited and games to write to csv
    games_df = createGamesPool()
    visited = []
    batch = []
    wrote_header_yet = False

    # Skip games already in the file and append to it instead of overwriting
    if resume:
        existing = pd.read_csv(filename, dtype={"GAME_ID": str})
        visited = existing["GAME_ID"].tolist()
        wrote_header_yet = True
        print(f"Resuming: {len(visited)} rows already in {filename}")
 
    for n in range(N):
        # training_df contains one row of training data
        training_df = createOneTrainingSet(games_df, visited)
        game_id = training_df["GAME_ID"].iloc[0]
        visited.append(game_id)
        batch.append(training_df)

        is_last_row = (n == N - 1)

        # Check if batch is long enough to save
        if len(batch) >= checkpoint_size or is_last_row:
            batch_df = pd.concat(batch, ignore_index=True)

            # If header is not present, write it
            if not wrote_header_yet:
                batch_df.to_csv(filename, index=False, mode="w", header=True)
                wrote_header_yet = True
            else:
                batch_df.to_csv(filename, index=False, mode="a", header=False)
            print(f"Checkpoint: wrote {len(batch)} rows ({n + 1}/{N} total so far)")
            batch = []

# createNTrainingSets(N=3000, filename="training_data_n=3000.csv")
# createNTrainingSets(N=600, filename="training_data_n=3000.csv", resume=True)
# createNTrainingSets(N=1000, filename="training_data_n=1000.csv")