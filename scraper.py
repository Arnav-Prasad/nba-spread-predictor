import pandas as pd
import time
from nba_api.stats.static import teams
from nba_api.stats.endpoints import teamgamelog, teamgamelogs


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
            temp = teamgamelog.TeamGameLog(
                team_id=team_id,
                season=season,
                season_type_all_star="Regular Season",
            )
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
            temp = teamgamelogs.TeamGameLogs(
                team_id_nullable=team_id,
                season_nullable=season,
                season_type_nullable="Regular Season",
                measure_type_player_game_logs_nullable="Advanced",
                )
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

        # Scrape game log for team, then convert to dataframe, add start and end dates
        seasons = ["2023-24", "2024-25", "2025-26"]
        games = []
        for season in seasons:
            temp = teamgamelog.TeamGameLog(
                team_id=team_id,
                season=season,
                season_type_all_star="Regular Season",
                date_from_nullable=start,
                date_to_nullable=end
            )
            temp = temp.get_data_frames()[0]
            games.append(temp)
            time.sleep(1)
        games_df = pd.concat(games, ignore_index=True)

        # Convert to datetime, sort the games chronologically
        games_df["GAME_DATE"] = pd.to_datetime(
            games_df["GAME_DATE"], 
            format="%b %d, %Y"
            )
        games_df = games_df.sort_values(by="GAME_DATE", ascending=False)

        # Add advanced stats like Net Rating
        advanced_games = []
        for season in seasons:
            temp = teamgamelogs.TeamGameLogs(
                team_id_nullable=team_id,
                season_nullable=season,
                season_type_nullable="Regular Season",
                measure_type_player_game_logs_nullable="Advanced",
                date_from_nullable=start,
                date_to_nullable=end
            )
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
team = Scraper("Los Angeles Lakers")
# print(team.lastNGames(5))
print(team.gamesDateRange(end="2026-04-12", start="2026-03-05"))
# print(team.averageStats(team.gamesDateRange(end="2026-04-12", start="2026-03-05")))