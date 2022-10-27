import numpy as np
import pandas as pd
import networkx as nx
import config
import operator
from sklearn.preprocessing import KBinsDiscretizer
from operator import itemgetter

from cleaner import clean




class Rank:
  """
  TODO: This should accept any pairwise statistics, not just points allowed/scored
  TODO: Investigate alternatives:
    -- Katz Centrality
    -- Temporal Page Rank: http://users.ics.aalto.fi/gionis/temporal-pagerank.pdf
    -- Temporal HITS? -> can't find any examples of this
    -- BeatPower
    -- http://snap.stanford.edu/class/cs224w-2019/slides/16-evolution.pdf
  """
  def __init__(self) -> None:
    self.read_path = './cleaned.csv'
    self.span = config.GAME_SPAN
    self.decay_span = 9 # self.span
    self.decay = 1 - 1 / self.decay_span

    data: pd.DataFrame = pd.read_csv(self.read_path)
    data.sort_values(['year', 'week'], inplace=True)

    self.data = data.drop(columns=[data.columns[0]]).reset_index(drop=True)

    # Array of each week's end index
    self.week_end_index = np.cumsum(data.groupby(['year', 'week']).size().to_list())

    # Array of each week's start index
    self.week_start_index = np.concatenate(([0], self.week_end_index[:-1]))

    # Initialize the points grid with scores from first [GAME_SPAN] weeks
    self.points_grid = self.init_grid()

    # Our initial Hits scores
    G = nx.from_pandas_adjacency(self.points_grid, create_using=nx.MultiDiGraph)
    start_week_points_allowed, start_week_points_for = nx.hits(G)
    self.p_a = self.rank(start_week_points_allowed)
    self.p_f = self.rank(start_week_points_for)


  def empty_grid(self):
    return pd.DataFrame(np.zeros((32, 32)), columns=config.FRANCHISE_ABBRV, index=config.FRANCHISE_ABBRV)


  def init_grid(self):
    """
    Initialize a 32 by 32 adj-mat with cumsum of scores against each opponent
    Ideally we'd do a 3D dataframe with time (week) as the z dimension, but pandas deprecated that. 
    """
    grid = self.empty_grid()

    for i in range(0, self.week_end_index[self.span]):
      row = self.data.iloc[i]
      grid.loc[row['Home_Team'], row['Away_Team']] += row['Away_Score'] * self.decay
      grid.loc[row['Away_Team'], row['Home_Team']] += row['Home_Score'] * self.decay

    return grid.round(3).clip(0)

  def update_grid(self, add_start, add_end, del_start, del_end):
    """
    Remove the entries right before the beginning of the window

    ~~~Updates are in place~~~

    We need to make updates on a weekly basis but since pandas doesn't allow for 3D dataframe I'm 
    relying on row indices.
    Not straightforward to use a numpy 3D because we need to maintain column and row names
    for the nx.DiGraph.
    """
    if del_start < 0:
      raise Exception("Cannot index less than 0 - check start/end bounds")

    # Compounded decay factor
    accumulated_decay = self.decay ** (self.decay_span + 1)

    for i in range(del_start, del_end):
      row = self.data.iloc[i]
      self.points_grid.loc[row['Home_Team'], row['Away_Team']] -= row['Away_Score'] * accumulated_decay
      self.points_grid.loc[row['Away_Team'], row['Home_Team']] -= row['Home_Score'] * accumulated_decay

    for i in range(add_start, add_end):
      row = self.data.iloc[i]
      self.points_grid.loc[row['Home_Team'], row['Away_Team']] += row['Away_Score']
      self.points_grid.loc[row['Away_Team'], row['Home_Team']] += row['Home_Score']

    self.points_grid *= self.decay

    # Because we accumulate rounding errors and get very small values around 0
    self.points_grid = self.points_grid.round(3).clip(0)

  def order(self, rank_dict, rank):
    """
    Order teams by rank
    """
    ordered = np.array([rank_dict[k] for k in rank])
    return ordered

  def rank(self, rank_dict):
    """
    Rank teams by HITS score
    """
    rank = [(k, v) for (k, v) in sorted(rank_dict.items(), key=itemgetter(1))]
    return rank

  def rolling_grid(self, add_start, add_end, del_start, del_end):
    """
    Make graph of the teams
    Arrows towards a team represent points scored
    Arrows away represent points allowed
    Run HITS
    Update the grid with current week's rankings
    """
    G = nx.from_pandas_adjacency(self.points_grid, create_using=nx.MultiDiGraph)
    points_allowed, points_for = nx.hits(G)
    self.update_grid(add_start=add_start, add_end=add_end, del_start=del_start, del_end=del_end)

    return points_allowed, points_for


  def hits_test(self):
    """
    Iterate through each year and week
    Update self.data with last values from self.p_a/self.p_f
    Record start and end indices for current week
    Make updates to the 32x32 grid that records each teams offensive/defensive strength in relation to each other
    Dump everything back into .csv
    """
    # clean(2022, 6, game_span=1)
    groups = self.data.groupby(['year', 'week'])

    # Make updates on a weekly basis
    for i, (k, v) in enumerate(groups):
      # Because we initialize grid with first [GAME_SPAN] weeks
      if k[0] == config.START_YEAR and k[1] < self.span + 1:
        continue
      if i >= len(groups) - 1:
        # Make a final update to .csv in the last iteration
        year, week = k
        for j, row in v.iterrows():
          self.fill_team_ranks(year, week, row)
        continue

      # Filling in the current week with values from previous week
      # This gives us trailing scores
      year, week = k
      for j, row in v.iterrows():
        self.fill_team_ranks(year, week, row)

      # Indices of upcoming week
      add_start = self.week_end_index[i]
      add_end = self.week_end_index[i + 1]

      # Need to subtract the scores from just before the game span
      # e.g. if game_span == 10, then on week 11 we subtract game 1
      del_start = self.week_start_index[i - self.span]
      del_end = self.week_start_index[i - (self.span - 1)]

      # Update grid with the current week
      p_a, p_f = self.rolling_grid(add_start, add_end, del_start, del_end)
      self.p_a = self.rank(p_a)
      self.p_f = self.rank(p_f)
      
    self.data.to_csv('./cleaned_with_rank.csv')

  def fill_team_ranks(self, year, week, row):
    """
    Updates each game in the dataframe with the home and away teams offense/defense ranks
    
    ~~~Updates are in place~~~
    """
    home_team = row['Home_Team']
    away_team = row['Away_Team']

    week_and_year_filter = (self.data['year'] == year) & (self.data['week'] == week)

    home_filter = (self.data['Home_Team'] == home_team)
    away_filter = (self.data['Away_Team'] == away_team)

    self.data.loc[week_and_year_filter & home_filter, 'Home_Off_Rank'] = [r[0] for r in self.p_f].index(home_team)
    self.data.loc[week_and_year_filter & home_filter, 'Home_Def_Rank'] = [r[0] for r in self.p_a].index(home_team)
    self.data.loc[week_and_year_filter & away_filter, 'Away_Off_Rank'] = [r[0] for r in self.p_f].index(away_team)
    self.data.loc[week_and_year_filter & away_filter, 'Away_Def_Rank'] = [r[0] for r in self.p_a].index(away_team)

  def idea(self):
    """
    Use PageRank/HITS to measure relative team strength
    Create a MultiDiGraph from the pairwise statistics of each team
    E.g., in a 2 node graph for NO vs PHL, we may have final score of 48-22
    Draw an arrow from PHL to NO with value of 48
    Draw an arrow from NO to PHL with value of 22
    First we create a matrix of pairwise statistics (scores, in this case)
    For each week, add scores to matrix (find a good rolling window (10?) (ewm?))
    Calculate HITS
    Replace the original team stat with it's HITS score
        NO PHL ATL CAR  PA
    NO  0   22  50  23  95   # PHL scored 22 against NO, ATL 50 against NO, CAR 23 against NO
    PHL 48  0   7   27  82    # NO 48 against PHL, ATL 7 against PHL, CAR 27 against PHL
    ATL 61  34  0   20  115   # NO 61 against ATL, PHL, 34 against ATL, CAR 20 against ATL
    CAR 10  59  28  0   97   # NO 40 against CAR, PHL 38 against CAR, ATL 47 against CAR
    PF  119 115 85  70
    T1 -> NO vs PHL: 48 - 22 | ATL vs CAR: 28 - 20
    T2 -> NO vs ATL: 35 - 27 | PHL vs CAR: 38 - 10
    T3 -> NO vs CAR: 10 - 23 | PHL vs ATL: 34 - 7
    T4 -> NO vs ATL: 26 - 23 | PHL vs CAR: 21 - 17
    """

  # Time stepped adjacency matrices -> S == T4
  # T(i) represents week i
    S = np.array([[0, 22, 50, 23],
                [48, 0, 7, 27],
                [61, 34, 0, 20],
                [10, 59, 28, 0]])
    T1 = np.array([[0, 22, 0, 0],
                [48, 0, 0, 0],
                [0, 0, 0, 20],
                [0, 0, 28, 0]])
    T2 = np.array([[0, 22, 27, 0],
                [48, 0, 0, 10],
                [35, 0, 0, 20],
                [0, 38, 28, 0]])
    T3 = np.array([[0, 22, 27, 23],
                [48, 0, 7, 10],
                [35, 34, 0, 20],
                [10, 38, 28, 0]])
    T4 = np.array([[0, 22, 50, 23],
                [48, 0, 7, 27],
                [61, 34, 0, 20],
                [10, 59, 28, 0]])

    G = nx.from_numpy_array(T4, create_using=nx.MultiDiGraph)
    pr = nx.pagerank(G) # Google's pagerank -> aggregates ingoing/outgoing (e.g. defense/offense is one score) - not as useful
    h, a = nx.hits(G) # Hits -> incoming and outgoing links are distinct (e.g. defense/offense are separate scores)
    # Time component (week to week delta) is difficult to encode but would be improvement
    print(pr) 
    print('offense') # higher better
    print(a)
    print('defense') # lower better
    print(h)

if __name__=='__main__':
  rank = Rank()
  # ~10 seconds
  rank.hits_test()