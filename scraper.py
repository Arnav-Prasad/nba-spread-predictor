import pandas as pd
import time
from nba_api.stats.static import teams
from nba_api.stats.endpoints import teamgamelog


class Scraper:

    # Intialize Scraper object
    def __init__(self, team):
        self.team = team

    '''
    Find team rolling stats across the last N games

    @param N    number of previous games to look at
    @return     Nested DataFrame of stats from last N games
    @author     Arnav Prasad
    @version    Sep 17, 2026
    '''
    def getTeamStats(self, N):
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

        # Convert to datetime, sort the games chronologically then return the first N
        games_df["GAME_DATE"] = pd.to_datetime(
            games_df["GAME_DATE"], 
            format="%b %d, %Y"
            )
        games_df = games_df.sort_values(by="GAME_DATE", ascending=False)
        print(games_df["GAME_DATE"].iloc[0])
        return games_df.head(N)

team = Scraper("Los Angeles Lakers")
print(team.getTeamStats(5))