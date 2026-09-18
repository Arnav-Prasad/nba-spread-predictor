import pandas as pd
from nba_api.stats.static import teams
from nba_api.stats.endpoints import teamgamelog, leaguedashteamstats, teamgamelogs


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
        team = next(team for team in teams.get_teams() if team['full_name'] == self.team)
        team_id = team['id']

        # Scrape game log for team, then convert to dataframe
        games = teamgamelog.TeamGameLog(
            team_id=team_id,
            season="2025-26",
            season_type_all_star="Regular Season",
        )
        games_df = games.get_data_frames()[0]

        # Convert to datetime, sort the games chronologically and limit to size N
        games_df["GAME_DATE"] = pd.to_datetime(
            games_df["GAME_DATE"], 
            format="%b %d, %Y"
            )
        games_df = games_df.sort_values(by="GAME_DATE", ascending=False)
        games_df = games_df.head(N)

        # Add advanced stats like Net Rating
        advanced_stats = teamgamelogs.TeamGameLogs(
            team_id_nullable=team_id,
            season_nullable="2025-26",
            season_type_nullable="Regular Season",
            measure_type_player_game_logs_nullable="Advanced"
        )
        # Sort the advanced stats, same as above
        advanced_df = advanced_stats.get_data_frames()[0]
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
    Calculate the averages of the games input. Averages of "OREB", "DREB",
    "REB", "AST", "STL", "BLK", "TOV", "PF", "PTS", "NET_RATING", "PACE", 
    "TM_TOV_PCT", "EFG_PCT", "TS_PCT", "AST_PCT", "AST_TO", "OREB_PCT", 
    "DREB_PCT", "REB_PCT", and "PIE" are calculated.

    @param games_df    DataFrame of stats from games to calculate averages
    @return     Dictionary with metrics as keys and averages as values
    @author     Arnav Prasad
    @version    Sep 17, 2026
    '''
    def averageLastNStats(self, games_df):
        # Intialize the averages dictionary with null values with each metric
        averages = {
                    "OREB": None, "DREB": None, "REB": None, "AST": None, 
                    "STL": None, "BLK": None, "TOV": None, "PF": None, 
                    "PTS": None, "NET_RATING": None, "PACE": None, 
                    "TM_TOV_PCT": None, "EFG_PCT": None, "TS_PCT": None, 
                    "AST_PCT": None, "AST_TO": None, "OREB_PCT": None, 
                    "DREB_PCT": None, "REB_PCT": None, "PIE": None
                    }

        # Iterate over averages and add the average metrics across the last N games
        # to the values of averages 
        for metric in averages.keys():
            averages[metric] = games_df[metric].mean(axis='rows')

        # averages now holds the required mean metrics
        return averages

    def gameRanges(self, start, end):
        # Find team Id
        team = next(team for team in teams.get_teams() if team['full_name'] == self.team)
        team_id = team['id']

        # Scrape game log for team, then convert to dataframe
        games = leaguedashteamstats.LeagueDashTeamStats(
            measure_type_detailed_defense="Advanced",
            season="2025-26",
            season_type_all_star="Regular Season",
            date_from_nullable=start,
            date_to_nullable=end
        )
        games_df = games.get_data_frames()[0]

        metrics_of_interest = [
                            "Team_ID", "Game_ID", "GAME_DATE", 
                            "MATCHUP", "WL", "W", "L", "W_PCT",
                            "MIN", "OREB", "DREB", "REB", "AST", 
                            "STL", "BLK", "TOV", "PF", "PTS", 
                            "NET_RATING", "PACE", "TM_TOV_PCT",
                            "EFG_PCT", "TS_PCT", "AST_PCT", "AST_TO",
                            "OREB_PCT", "DREB_PCT", "REB_PCT", "PIE"
                            ]
        # games_df = games_df[metrics_of_interest]
        print(games_df.columns)
        return games_df

# Tests--
team = Scraper("Los Angeles Lakers")
print(team.lastNGames(5))
print(team.gameRanges(end="2026-04-12", start="2026-04-05"))