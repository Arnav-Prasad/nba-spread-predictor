import pandas as pd
import time
from nba_api.stats.static import teams
from nba_api.stats.endpoints import teamgamelog


class Scraper:

    # Intialize Scraper object
    def __init__(self, team1, team2):
        self.team1 = team1
        self.team2 = team2

    '''
    Find team1 rolling stats across the last N games

    @param N    number of previous games to look at
    @return     Nested DataFrame of stats from last N games
    '''
    def getTeam1Stats(self, N):
        # Find team1Id
        team = next(team for team in teams.get_teams() if team['full_name'] == self.team1)
        team_id = team['id']

        # Scrape game log for team1, then convert to dataframe
        games = teamgamelog.TeamGameLog(
            team_id=team_id,
            season="2025-26",
            season_type_all_star="Regular Season",
        )
        games_df = games.get_data_frames()[0]

        # Sort the games chronologically then return the first N
        return games_df.sort_values(by="GAME_DATE", ascending=False).head(N)

    '''
        Find team2 rolling stats across the last N games
    
        @param N    number of previous games to look at
        @return     Nested DataFrame of stats from last N games
    '''
    def getTeam2Stats(self, N):
        # Find team2Id
        team = next(team for team in teams.get_teams() if team['full_name'] == self.team2)
        team_id = team['id']

        # Scrape game log for team2, then convert to dataframe
        games = teamgamelog.TeamGameLog(
            team_id=team_id,
            season="2025-26",
            season_type_all_star="Regular Season",
        )
        games_df = games.get_data_frames()[0]

        # Sort the games chronologically then return the first N
        return games_df.sort_values(by="GAME_DATE", ascending=False).head(N)
