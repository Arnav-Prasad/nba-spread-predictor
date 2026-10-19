import pandas as pd
import time
from nba_api.stats.static import teams
from nba_api.stats.endpoints import teamgamelog, teamgamelogs, leaguegamefinder

MAX_REST = 5  # 4 days off and 9 days off aren't meaningfully different


'''
Call fetch_func, retrying up to max_retries times with a delay in between
if it raises an exception (e.g. a network timeout). Used to wrap nba_api
calls so a transient server hiccup doesn't crash the whole script.

@param fetch_func     A zero-argument function (e.g. a lambda) that makes
                       the actual API call and returns its result
@param max_retries     How many total attempts to make before giving up
@param delay           Seconds to wait between retries
@return                Whatever fetch_func() returns, on the first success
@author                Arnav Prasad
@version               Sep 22, 2026
'''
def fetchWithRetry(fetch_func, max_retries=3, delay=5):
    for attempt in range(max_retries):
        try:
            return fetch_func()
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(delay)
    raise Exception("Max retries exceeded")


'''
Pull every Regular Season game for the given seasons, including each
season's first 3 weeks (a game just past the training cutoff can have its
previous game inside that window). One row per team per game. Used to
compute rest days.

@param seasons    list of season strings, e.g. "2023-24"
@return           DataFrame with GAME_ID, TEAM_ID, SEASON_ID, GAME_DATE, MATCHUP
@author           Arnav Prasad
@version          Oct 14, 2026
'''
def fetchAllGames(seasons=("2023-24", "2024-25", "2025-26")):
    games = []
    for season in seasons:
        temp = fetchWithRetry(lambda season=season: leaguegamefinder.LeagueGameFinder(
            league_id_nullable="00",
            season_type_nullable="Regular Season",
            season_nullable=season
        ))
        games.append(temp.get_data_frames()[0])
        time.sleep(1)
    games_df = pd.concat(games, ignore_index=True)
    games_df["GAME_DATE"] = pd.to_datetime(games_df["GAME_DATE"])
    games_df["GAME_ID"] = games_df["GAME_ID"].astype(str)
    return games_df[["GAME_ID", "TEAM_ID", "SEASON_ID", "GAME_DATE", "MATCHUP"]]


'''
Compute rest_diff for every game: home team's days of rest minus the away
team's. Rest = days since that team's previous game in the same season
(1 = back-to-back), capped at MAX_REST. A team's first game of a season
gets MAX_REST.

@param games_df    output of fetchAllGames
@return            Series of rest_diff, indexed by GAME_ID
@author            Arnav Prasad
@version           Oct 14, 2026
'''
def computeRestDiff(games_df):
    games_df = games_df.drop_duplicates(subset=["TEAM_ID", "GAME_ID"])
    games_df = games_df.sort_values(by=["TEAM_ID", "GAME_DATE"])

    # Days since the same team's previous game, within the same season
    games_df["rest"] = games_df.groupby(["TEAM_ID", "SEASON_ID"])["GAME_DATE"].diff().dt.days
    games_df["rest"] = games_df["rest"].fillna(MAX_REST).clip(upper=MAX_REST)

    # "@" in the matchup means this row is the away team
    is_home = -(games_df["MATCHUP"].str.contains("@")+1)
    home = games_df[is_home].set_index("GAME_ID")["rest"]
    away = games_df[~is_home].set_index("GAME_ID")["rest"]

    rest_df = pd.DataFrame({"home_rest": home, "away_rest": away}).dropna()
    return (rest_df["home_rest"] - rest_df["away_rest"]).rename("rest_diff")


class Scraper:

    # Intialize Scraper object
    def __init__(self, team):
        self.team = team

    '''
    Find team rolling stats across the last N games. Specifically gets
    "Team_ID", "Game_ID", "GAME_DATE", "MATCHUP", "WL", "W", "L", "W_PCT",
    "MIN", "OREB", "DREB", "REB", "AST", "STL", "BLK", "TOV", "PF", "PTS", 
    "NET_RATING", "PACE", "TM_TOV_PCT", "EFG_PCT", "TS_PCT", "AST_PCT", 
    "AST_TO", "OREB_PCT", "DREB_PCT", "REB_PCT", and "PIE".

    @param N    number of previous games to look at
    @return     Nested DataFrame of stats from last N games
    @author     Arnav Prasad
    @version    Sep 17, 2026
    '''
    def lastNGames(self, N):
        # Find team Id
        team = next(t for t in teams.get_teams() if t['full_name'] == self.team)
        team_id = team["id"]

        # Scrape game log for team, then convert to dataframe
        seasons = ["2023-24", "2024-25", "2025-26"]
        games = []
        for season in seasons:
            temp = fetchWithRetry(lambda season=season: teamgamelog.TeamGameLog(
                team_id=team_id,
                season=season,
                season_type_all_star="Regular Season",
            ))
            temp = temp.get_data_frames()[0]
            games.append(temp)
            time.sleep(1)
        games_df = pd.concat(games, ignore_index=True)

        # Convert to datetime, sort the games chronologically and limit to size N
        games_df["GAME_DATE"] = pd.to_datetime(
            games_df["GAME_DATE"],
            format="%b %d, %Y"
            )
        games_df = games_df.sort_values(by="GAME_DATE", ascending=False)
        games_df = games_df.head(N)

        # Add advanced stats like Net Rating
        advanced_games = []
        for season in seasons:
            temp = fetchWithRetry(lambda season=season: teamgamelogs.TeamGameLogs(
                team_id_nullable=team_id,
                season_nullable=season,
                season_type_nullable="Regular Season",
                measure_type_player_game_logs_nullable="Advanced",
                ))
            temp = temp.get_data_frames()[0]
            advanced_games.append(temp)
            time.sleep(1)
        advanced_df = pd.concat(advanced_games, ignore_index=True)

        # Sort the advanced stats, same as above
        advanced_df["GAME_DATE"] = pd.to_datetime(
            advanced_df["GAME_DATE"],
            format="%Y-%m-%dT%H:%M:%S"
            )
        advanced_df = advanced_df.sort_values(by="GAME_DATE", ascending=False)
        advanced_df = advanced_df.head(N)

        # Merge the two DataFrames into one
        metrics_of_interest = [
                            "NET_RATING", "PACE", "TM_TOV_PCT", "EFG_PCT",
                            "TS_PCT", "AST_PCT", "AST_TO", "OREB_PCT",
                            "DREB_PCT", "REB_PCT", "PIE"
                            ]
        games_df = games_df.merge(
                        advanced_df.set_index("GAME_ID")[metrics_of_interest],
                        left_on="Game_ID",
                        right_index=True,
                        how="left",
                        )

        # games_df now holds all the needed data
        return games_df

    '''
    Find team rolling stats across a given calendar range. Specifically gets
    "Team_ID", "Game_ID", "GAME_DATE", "MATCHUP", "WL", "W", "L", "W_PCT",
    "MIN", "OREB", "DREB", "REB", "AST", "STL", "BLK", "TOV", "PF", "PTS", 
    "NET_RATING", "PACE", "TM_TOV_PCT", "EFG_PCT", "TS_PCT", "AST_PCT", 
    "AST_TO", "OREB_PCT", "DREB_PCT", "REB_PCT", and "PIE" between two dates.

    @param start    date to begin looking at games, inclusive
    @param end      date to end looking at games, inclusive
    @return         Nested DataFrame of stats from given date range
    @author         Arnav Prasad
    @version        Sep 18, 2026
    '''
    def gamesDateRange(self, start, end):
        # Find team Id
        team = next(t for t in teams.get_teams() if t['full_name'] == self.team)
        team_id = team["id"]

        # Get which season start is in
        start_date = pd.to_datetime(start, format="%Y-%m-%d")
        if (start_date > pd.to_datetime("2025-09-01", format="%Y-%m-%d")):
            season = "2025-26"
        elif (start_date > pd.to_datetime("2024-09-01", format="%Y-%m-%d")):
            season = "2024-25"
        elif (start_date > pd.to_datetime("2023-09-01", format="%Y-%m-%d")):
            season = "2023-24"
        else:
            raise ValueError(f"No season mapping for {start_date}")

        # Scrape game log for team, then convert to dataframe, add start and end dates
        temp = fetchWithRetry(lambda: teamgamelog.TeamGameLog(
            team_id=team_id,
            season=season,
            season_type_all_star="Regular Season",
            date_from_nullable=start,
            date_to_nullable=end
        ))
        games_df = temp.get_data_frames()[0]
        time.sleep(1)

        # Convert to datetime, sort the games chronologically
        games_df["GAME_DATE"] = pd.to_datetime(
            games_df["GAME_DATE"],
            format="%b %d, %Y"
            )
        games_df = games_df.sort_values(by="GAME_DATE", ascending=False)

        # Add advanced stats like Net Rating
        temp = fetchWithRetry(lambda: teamgamelogs.TeamGameLogs(
            team_id_nullable=team_id,
            season_nullable=season,
            season_type_nullable="Regular Season",
            measure_type_player_game_logs_nullable="Advanced",
            date_from_nullable=start,
            date_to_nullable=end
        ))
        advanced_df = temp.get_data_frames()[0]
        time.sleep(1)

        # Sort the advanced stats, same as above
        advanced_df["GAME_DATE"] = pd.to_datetime(
            advanced_df["GAME_DATE"],
            format="%Y-%m-%dT%H:%M:%S"
            )
        advanced_df = advanced_df.sort_values(by="GAME_DATE", ascending=False)

        # Merge the two DataFrames into one
        metrics_of_interest = [
                            "NET_RATING", "PACE", "TM_TOV_PCT", "EFG_PCT",
                            "TS_PCT", "AST_PCT", "AST_TO", "OREB_PCT",
                            "DREB_PCT", "REB_PCT", "PIE"
                            ]
        games_df = games_df.merge(
                        advanced_df.set_index("GAME_ID")[metrics_of_interest],
                        left_on="Game_ID",
                        right_index=True,
                        how="left",
                    )

        # games_df now holds all the needed data
        return games_df

    '''
    Calculate the averages of the games input. Averages of "OREB", "DREB",
    "REB", "AST", "STL", "BLK", "TOV", "PF", "PTS", "NET_RATING", "PACE", 
    "TM_TOV_PCT", "EFG_PCT", "TS_PCT", "AST_PCT", "AST_TO", "OREB_PCT", 
    "DREB_PCT", "REB_PCT", and "PIE" are calculated.

    @param games_df    DataFrame of stats from games to calculate averages
    @return            DataFrame of metrics and averages values
    @author            Arnav Prasad
    @version           Sep 17, 2026
    '''
    def averageStats(self, games_df):
        # Intialize the metrics list
        metrics_of_interest = [
                    "OREB", "DREB", "REB", "AST", "STL", "BLK", "TOV", "PF", "PTS", 
                    "NET_RATING", "PACE", "TM_TOV_PCT", "EFG_PCT", "TS_PCT", 
                    "AST_PCT", "AST_TO", "OREB_PCT", "DREB_PCT", "REB_PCT", "PIE"
                ]

        # Take the average of the metrics in games_df that is in common with metrics_of_interest
        averages_df = games_df[metrics_of_interest].mean().to_frame().T

        # average_df now holds the mean metrics in DataFrame form
        return averages_df

# Tests--
# team = Scraper("Los Angeles Lakers")
# print(team.lastNGames(5))
# print(team.gamesDateRange(end="2026-04-12", start="2026-03-05"))
# print(team.averageStats(team.gamesDateRange(end="2026-04-12", start="2026-03-05")))