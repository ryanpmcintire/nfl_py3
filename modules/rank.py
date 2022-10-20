import numpy as np
import pandas as pd
import networkx as nx
import config
import operator

from cleaner import clean

read_path = './cleaned_with_dupes.csv'



def empty_grid():
  return pd.DataFrame(np.zeros((32, 32)), columns=config.FRANCHISE_ABBRV, index=config.FRANCHISE_ABBRV)


def init_grid(data: pd.DataFrame, window=10):
  """
  Initialize a 32 by 32 adj-mat with cumsum of scores against each opponent
  """
  grid = empty_grid()
  for i in range(0, window*16):
    row = data.iloc[i]
    grid.loc[row['Home_Team'], row['Away_Team']] += row['Away_Score']
    grid.loc[row['Away_Team'], row['Home_Team']] += row['Home_Score']
  return grid

def update_grid(data, grid, start=0, size = 16, window=10):
  """
  Remove the entries right before the beginning of the window
  This is 16 rows -> 1 for each game
  Updates are in place
  """
  del_start = start - window * 16
  if del_start < 0:
    raise Exception("Cannot index less than 0 - check start and window bounds")
  del_end = del_start + size
  for i in range(del_start, del_end):
    row = data.iloc(i)
    print(row)
    grid.loc[row['Home_Team'], row['Away_Team']] -= row['Away_Score']
    grid.loc[row['Away_Team'], row['Home_Team']] -= row['Home_Score']

  # G = nx.from_pandas_adjacency(e, create_using=nx.MultiDiGraph)

  # points_allowed, points_for = nx.hits(G)

  # print('offense') # higher better
  # print(points_for)
  # print('defense') # lower better
  # print(points_allowed)

  # rank_a = rank(points_for)
  # ordered_a = order(points_for, rank_a)
  # print(rank_a)
  # print(ordered_a)
  # rank_d = rank(points_allowed)
  # ordered_d = order(points_allowed, rank_d)
  # print(rank_d)
  # print(ordered_d)

def order(rank_dict, rank):
    ordered = np.array([rank_dict[k] for k in rank])
    return ordered

def rank(rank_dict):
    rank = [k for (k, v) in sorted(rank_dict.items(), key=operator.itemgetter(1))]
    return rank

def rolling_grid(data, grid, start, size):
  G = nx.from_pandas_adjacency(grid, create_using=nx.MultiDiGraph)
  update_grid(data=data, grid=grid, start=start, size=size)
  points_allowed, points_for = nx.hits(G)
  return points_allowed, points_for


def hits_test():
  # clean(2022, 6, game_span=1)
  data: pd.DataFrame = pd.read_csv(read_path)
  data.sort_values(['year', 'week'], inplace=True)
  points_grid = init_grid(data)
  rolling_hits = pd.DataFrame()

  start = 160
  for k, v in data.groupby(['year', 'week']):
    size = len(v)
    p_a, p_f = rolling_grid(data, points_grid, start, size)
    start += size
    print(p_a, p_f)
  


def idea():
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
  hits_test()