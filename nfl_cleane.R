# This script is for cleaning the data that was scraped from pro-football reference
# Need to do some spring-cleaning on this when 2018 season is over

# These two libs are only needed if run into trouble installing the below
# require(devtools)
# require(pkgbuild)

# Neccessary libs
require(purrr)
require(tidyr)
require(dplyr)
require(zoo)
require(caret)
require(mxnet)

# Double check the path
# Had to create a separate csv for the current season because of
# limitations with the python scraper that I didn't have time to fix.
# This separation ends up being convenient down the road, though
path <- "D:/nfl_py3/nfl_master_2009-2017.csv"
currentSeason <- "D:/nfl_py3/nfl_master_2018-2018.csv"
rawSet <- read.csv(path, stringsAsFactors = FALSE)
newSet <- read.csv(currentSeason, stringsAsFactors = FALSE)

# This is necessary because an incomplete season has only integers in week column
# a complete season has characters because of playoffs
newSet$week <- as.character(newSet$week)
rawSet <- bind_rows(rawSet, newSet)

# Rename columns because I messed up names in scraper
colnames(rawSet)[6:7] <- c("boxscoreUri", "time")

# Get rid of playoffs and convert weeks to integer
# for now, only look at regular season. Playoffs are unique for lots of reasons so will only add noise
playoffs <- c("Wild Card", "Division", "Conf. Champ.", "SuperBowl")
regSeason <- rawSet %>%
  filter(!(week %in% playoffs))
regSeason$week <- as.integer(regSeason$week)
# Test above code
stopifnot(n_distinct(regSeason$week) == 17)

# football-reference.com is inconsistent with naming of chargers and rams following their move to new cities
# need to add two more entries for these teams because of this
opponent_names <- regSeason %>%
  select(team) %>%
  unique()
opponent_names <- append(opponent_names$team, c("sdg", "ram"))

verbose_names <- regSeason %>%
  select(verbose_name) %>%
  unique()
verbose_names <- append(verbose_names$verbose_name, c("Los Angeles Chargers", "Los Angeles Rams"))

# Store name changes in intermediate matrix
name_changes <- c("sdg", "ram", "Los Angeles Chargers", "Los Angeles Rams")
mat <- matrix(name_changes, nrow = 2, ncol = 2)
colnames(mat) <- c("opponent_names", "verbose_names")

# Create lookup table and replace opponent names with 3 letter abbrv
# This makes it easier to treat the San Diego Chargers as the same
# team as the Los Angeles Chargers. Likewise for Rams. We use the
# three letter abbrv for other things down the road.
lookup <- data.frame(opponent_names, verbose_names)
lookup <- rbind(lookup, mat)
regSeason$opponent <- lookup$opponent_names[match(unlist(regSeason$opponent), lookup$verbose_names)]
# Test above code
stopifnot(n_distinct(regSeason$opponent) == 32)

# If need to pull in upcoming week, use below to create a sheet with proper formatting,
# colnames etc. Enter upcoming week's speads from preferred source.
# Then read in the sheet with the upcoming week's matchups.
# This is really the only part of the process that I haven't completely automated yet :(
# Don't mess this part up or the data will all be misaligned down the road.
currentYear <- regSeason %>%
  filter(year == 2018)
write.csv(currentYear, file = "nfl_newWeek.csv", row.names = FALSE)
# Import upcoming week
newWeek <- read.csv("C:/Users/Ryan/Downloads/nfl_newWeek16.csv", stringsAsFactors = FALSE)

# Bind the above .csv
regSeason <- regSeason %>%
  filter(year < 2018)
regSeason <- bind_rows(regSeason, newWeek)

# The data pulled from football-reference.com is in a game-instance format.
# This is extremely inconvenient to work with, so we have to reorder
# everything to be in a team-game-instance format. This will make it
# possible to create trailing stats for each team. We end up duplicating game-instance
# records by doing this but that is trivial to deal with later.
# First step is figure out who was home team/away team for each game
regSeason$Home_Team <- with(regSeason, ifelse(at == "@", opponent, team))
regSeason$Away_Team <- with(regSeason, ifelse(at == "@", team, opponent))

# Create home and away score cols
regSeason$Home_Score <- with(regSeason, ifelse(at == "@", opp_score, team_score))
regSeason$Away_Score <- with(regSeason, ifelse(at == "@", team_score, opp_score))

# Remove city prefixes so parsing vegas line is easier
regSeason$Vegas_Line_Close <- sub("^(San |New |St. |Green |Tampa |Kansas |Los )*", "", regSeason$Vegas_Line_Close)

# Split the vegas line columns
vegas_line_split <- c("Fav_City", "Fav_Team", "Fav_Spread")
regSeason <- as.data.frame(regSeason) %>%
  separate(Vegas_Line_Close, into = vegas_line_split, sep = " ", remove = FALSE, fill = "right")
regSeason$Fav_Spread <- as.numeric(regSeason$Fav_Spread)

# Figure out if home team is favorite. 1 = favorite, 0 is pick, -1 = underdog.
# Ordering of short names is copy/pasted in order they first appear in data set.
# This particular order comes from 2009 season. This will break if first season in
# dataset is changed so leaving this here to catch that (helping out future Ryan)
stopifnot(regSeason$year[1] == 2009)
favorite <- regSeason %>%
  select(Fav_Team) %>%
  unique()
shortName <- c("crd", "jax", "sea", "nyg", "chi", "oti", "min", "atl",
               "sdg", "nor", "kan", "sfo", "ram", "den", "car", "dal",
               "was", "pit", "rav", "phi", "cin", "nwe", "gnb", "nyj",
               "det", "tam", "htx", "000", "mia", "buf", "rai", "clt", "cle")
favLookup <- data.frame(shortName, favorite)
regSeason$favorite <- as.character(favLookup$shortName[match(unlist(regSeason$Fav_Team), favLookup$Fav_Team)])
regSeason$HomeFav <- with(regSeason, ifelse(Home_Team == favorite, 1, ifelse(favorite == "000", 0, -1)))
regSeason$underdog <- with(regSeason, ifelse(team == favorite, 0, 1))

# Replace NA's for turnovers, spreads, fav_city.
regSeason <- regSeason %>% replace(., is.na(.), 0)

# Add in the spread with the sign negative if home team is fav, 0 if pick'em, positive if underdog
# Rearrange spread so it is in terms of the home team, rather than in terms of the favorite (vegas' format).
# We want our machine learning algorithms to be able to take advantage of the home-field phenomenon.
# So organizing everything in terms of the home team winning/losing makes this a bit easier I think.
# A pitfall of this: Games where neither team is home aren't accounted for (e.g. London games)
regSeason$Home_Vegas_Spread <- with(regSeason, ifelse(HomeFav == 1, Fav_Spread, ifelse(HomeFav == 0, 0, Fav_Spread * -1)))
regSeason$Home_Actual_Spread <- with(regSeason, -1 * (Home_Score - Away_Score))

# Storing some new column names by home/away
away_pass_stats <- c("aCmp", "aAtt", "aYd", "aTD", "aINT")
home_pass_stats <- c("hCmp", "hAtt", "hYd", "hTD", "hINT")
away_rush_stats <- c("aRush", "aRYds", "aRTDs")
home_rush_stats <- c("hRush", "hRYds", "hRTDs")
away_pen_yds <- c("aPen", "aPenYds")
home_pen_yds <- c("hPen", "hPenYds")
away_third_downs <- c("aThrdConv", "aThrd")
home_third_downs <- c("hThrdConv", "hThrd")
away_fourth_downs <- c("aFrthConv", "aFrth")
home_fourth_downs <- c("hFrthConv", "hFrth")

# Lots of ugly lines doing the same thing. Seemed like it wasn't a good use of time
# to turn this into functions when copy/paste was quick enough.
# Here I am creating some stats based on the offense/defense performance of the
# home and away teams.
# Passing columns
regSeason <- as.data.frame(regSeason) %>%
  separate(aCmp.Att.Yd.TD.INT, into = away_pass_stats, remove = FALSE) %>%
  separate(hCmp.Att.Yd.TD.INT, into = home_pass_stats, remove = FALSE)
regSeason$Cmp <- as.numeric(with(regSeason, ifelse(at == "@", aCmp, hCmp)))
regSeason$Att <- as.numeric(with(regSeason, ifelse(at == "@", aAtt, hAtt)))
regSeason$Yd <- as.numeric(with(regSeason, ifelse(at == "@", aYd, hYd)))
regSeason$TD <- as.numeric(with(regSeason, ifelse(at == "@", aTD, hTD)))
regSeason$INT <- as.numeric(with(regSeason, ifelse(at == "@", aINT, hINT)))
regSeason$dCmp <- as.numeric(with(regSeason, ifelse(at == "@", hCmp, aCmp)))
regSeason$dAtt <- as.numeric(with(regSeason, ifelse(at == "@", hAtt, aAtt)))
regSeason$dYd <- as.numeric(with(regSeason, ifelse(at == "@", hYd, aYd)))
regSeason$dTD <- as.numeric(with(regSeason, ifelse(at == "@", hTD, aTD)))
regSeason$dINT <- as.numeric(with(regSeason, ifelse(at == "@", hINT, aINT)))
regSeason$team_pass_eff <- regSeason$Yd / regSeason$Att
regSeason$team_pass_def <- regSeason$dYd / regSeason$dAtt
# Rushing columns
regSeason <- as.data.frame(regSeason) %>%
  separate(aRush.Yds.Tds, into = away_rush_stats, remove = FALSE) %>%
  separate(hRush.Yds.Tds, into = home_rush_stats, remove = FALSE)
regSeason$Rush <- as.numeric(with(regSeason, ifelse(at == "@", aRush, hRush)))
regSeason$RYds <- as.numeric(with(regSeason, ifelse(at == "@", aRYds, hRYds)))
regSeason$RTDs <- as.numeric(with(regSeason, ifelse(at == "@", aRTDs, hRTDs)))
regSeason$dRush <- as.numeric(with(regSeason, ifelse(at == "@", hRush, aRush)))
regSeason$dRYds <- as.numeric(with(regSeason, ifelse(at == "@", hRYds, aRYds)))
regSeason$dRTDs <- as.numeric(with(regSeason, ifelse(at == "@", hRTDs, aRTDs)))
regSeason$team_rush_eff <- regSeason$RYds / regSeason$Rush
regSeason$team_rush_def <- regSeason$dRYds / regSeason$dRush
# Penalties columns
regSeason <- as.data.frame(regSeason) %>%
  separate(aPenalties.Yds, into = away_pen_yds, remove = FALSE) %>%
  separate(h.Penalties.Yds, into = home_pen_yds, remove = FALSE)
regSeason$Pen <- as.numeric(with(regSeason, ifelse(at == "@", aPen, hPen)))
regSeason$PenYds <- as.numeric(with(regSeason, ifelse(at == "@", aPenYds, hPenYds)))
regSeason$PenAgg <- as.numeric(with(regSeason, ifelse(at == "@", hPen, aPen)))
regSeason$PenYdsAgg <- as.numeric(with(regSeason, ifelse(at == "@", hPenYds, aPenYds)))
# Down columns
regSeason <- as.data.frame(regSeason) %>%
  separate(aThird_Down_Conv, into = away_third_downs, remove = FALSE) %>%
  separate(hThird_Down_Conv, into = home_third_downs, remove = FALSE) %>%
  separate(aFourth_Down_Conv, into = away_fourth_downs, remove = FALSE) %>%
  separate(hFourth_Down_Conv, into = home_fourth_downs, remove = FALSE)
regSeason$ThrdConv <- as.numeric(with(regSeason, ifelse(at == "@", aThrdConv, hThrdConv)))
regSeason$dThrdConv <- as.numeric(with(regSeason, ifelse(at == "@", hThrdConv, aThrdConv)))
regSeason$Thrds <- as.numeric(with(regSeason, ifelse(at == "@", aThrd, hThrd)))
regSeason$dThrds <- as.numeric(with(regSeason, ifelse(at == "@", hThrd, aThrd)))
regSeason$FrthConv <- as.numeric(with(regSeason, ifelse(at == "@", aFrthConv, hFrthConv)))
regSeason$dFrthConv <- as.numeric(with(regSeason, ifelse(at == "@", hFrthConv, aFrthConv)))
regSeason$Frths <- as.numeric(with(regSeason, ifelse(at == "@", aFrth, hFrth)))
regSeason$dFrths <- as.numeric(with(regSeason, ifelse(at == "@", hFrth, aFrth)))
regSeason$third_eff <- regSeason$ThrdConv / regSeason$Thrds
regSeason$third_def <- regSeason$dThrdConv / regSeason$dThrds
regSeason$fourth_eff <- regSeason$FrthConv / regSeason$Frths
regSeason$fourth_def <- regSeason$dFrthConv / regSeason$dFrths
# We get some x / 0 in the fourths columns so must remove those
# Really need to introduce laplace transformation or something similar but this is good enough for now
regSeason$fourth_eff[is.nan(regSeason$fourth_eff)] <- 0.0001
regSeason$fourth_def[is.nan(regSeason$fourth_def)] <- 0.0001

# This line gets the col index of the opponent which we can then use later on for lookups and stuff
# Conveniently, the boxscoreUri is a unique identifier for each game
regSeason$opponent_col <- with(regSeason, ave(seq_along(boxscoreUri), boxscoreUri, FUN = rev))

# Calculate some revert to mean stats and streak stats
regSeason <- regSeason %>%
  group_by(team) %>%
  mutate(lostLastAsFav = ifelse(lag(favorite) == lag(team) & lag(result) == "L", 1, 0)) %>%
  mutate(wonLastAsDog = ifelse(lag(underdog) == 1 & lag(result) == "W", 1, 0))

# Game span stats
# A game span of 10 is chosen. There is no science to this other than a game span of 1 is too small and a game span
# of 20 is too large.
# Game spans of 8, 9, 11 produce similar enough results as game span of 10 but I didn't want to focus on optimizing this
# when the minor differences in performance are likely due to chance.
# However, it is probably worth investigating weighted moved averages, with the most recent game weighted the most, the
# oldest game weighted the least. That too, could get into overfitting territory if not careful, though.
game_span = 10
# Moving avgs up to but not including current game (by team)
# Including current game would introduce look-ahead bias. (I saw some people doing this in my google research :/ )
regSeason <- regSeason %>%
  group_by(team) %>%
  mutate(trail_score = rollapplyr(team_score, list(-(game_span:1)), mean, fill = NA)) %>%
  mutate(trail_allow = rollapplyr(opp_score, list(-(game_span:1)), mean, fill = NA)) %>%
  mutate(trail_to = rollapplyr(off_turn_overs, list(-(game_span:1)), mean, fill = NA)) %>%
  mutate(trail_fto = rollapplyr(def_turn_overs, list(-(game_span:1)), mean, fill = NA)) %>%
  mutate(trail_pass_eff = rollapplyr(team_pass_eff, list(-(game_span:1)), mean, fill = NA)) %>%
  mutate(trail_pass_def = rollapplyr(team_pass_def, list(-(game_span:1)), mean, fill = NA)) %>%
  mutate(trail_rush_eff = rollapplyr(team_rush_eff, list(-(game_span:1)), mean, fill = NA)) %>%
  mutate(trail_rush_def = rollapplyr(team_rush_def, list(-(game_span:1)), mean, fill = NA)) %>%
  mutate(trail_penYds = rollapplyr(PenYds, list(-(game_span:1)), mean, fill = NA)) %>%
  mutate(trail_penYdsAgg = rollapplyr(PenYdsAgg, list(-(game_span:1)), mean, fill = NA)) %>%
  mutate(trail_third_eff = rollapplyr(third_eff, list(-(game_span:1)), mean, fill = NA)) %>%
  mutate(trail_third_def = rollapplyr(third_def, list(-(game_span:1)), mean, fill = NA)) %>%
  mutate(trail_fourth_eff = rollapplyr(fourth_eff, list(-(game_span:1)), mean, fill = NA)) %>%
  mutate(trail_fourth_def = rollapplyr(fourth_def, list(-(game_span:1)), mean, fill = NA))
# Moving avgs up to but not including current game (by opponent)
regSeason <- regSeason %>%
  mutate(trail_opp_score = regSeason$trail_score[opponent_col]) %>%
  mutate(trail_opp_allow = regSeason$trail_allow[opponent_col]) %>%
  mutate(trail_opp_to = regSeason$trail_to[opponent_col]) %>%
  mutate(trail_opp_fto = regSeason$trail_fto[opponent_col]) %>%
  mutate(trail_opp_pass_eff = regSeason$trail_pass_eff[opponent_col]) %>%
  mutate(trail_opp_pass_def = regSeason$trail_pass_def[opponent_col]) %>%
  mutate(trail_opp_rush_eff = regSeason$trail_rush_eff[opponent_col]) %>%
  mutate(trail_opp_rush_def = regSeason$trail_rush_def[opponent_col]) %>%
  mutate(trail_opp_penYds = regSeason$trail_penYds[opponent_col]) %>%
  mutate(trail_opp_penYdsAgg = regSeason$trail_penYdsAgg[opponent_col]) %>%
  mutate(trail_opp_third_eff  = regSeason$trail_third_eff[opponent_col]) %>%
  mutate(trail_opp_third_def  = regSeason$trail_third_def[opponent_col]) %>%
  mutate(trail_opp_fourth_eff  = regSeason$trail_fourth_eff[opponent_col]) %>%
  mutate(trail_opp_fourth_def  = regSeason$trail_fourth_def[opponent_col])

# Here we are selecting the features we want to feed the machine algorithms.
# First four cols represent information to identify the games
# Remaining cols represent the trailing avgs of various statistics leading up to but not including the game
# Exception is the result col which becomes our prediction label (whether the home team wins/loses/ties)
# The per-team trailing stats we've created also need to have their context switched to home/away trailing stats
features <- data.frame("Year" = regSeason$year, "Week" = regSeason$week, "Home_Team" = regSeason$Home_Team,
                       "Away_Team" = regSeason$Away_Team)
features$Home_Win <- as.factor(with(regSeason,
                                    ifelse(result == "L" & at == "", "L",
                                           ifelse(result == "W" & at == "", "W",
                                                  ifelse(result == "L" & at == "@", "W", "L")))))
features$Home_Fav <- as.factor(regSeason$HomeFav)
features$Home_Vegas_Spread <- as.numeric(regSeason$Home_Vegas_Spread)
features$Home_Actual_Spread <- regSeason$Home_Actual_Spread
features$Home_Score <- regSeason$Home_Score
features$Away_Score <- regSeason$Away_Score
features$Trail_Home_Score <- with(regSeason, ifelse(at == "@", trail_opp_score, trail_score))
features$Trail_Away_Score <- with(regSeason, ifelse(at == "@", trail_score, trail_opp_score))
features$Home_Allowed <- with(regSeason, ifelse(at == "@", trail_opp_allow, trail_allow))
features$Away_Allowed <- with(regSeason, ifelse(at == "@", trail_allow, trail_opp_allow))
features$Home_TO <- with(regSeason, ifelse(at == "@", trail_opp_to, trail_to))
features$Away_TO <- with(regSeason, ifelse(at == "@", trail_to, trail_opp_to))
features$Home_FTO <- with(regSeason, ifelse(at == "@", trail_opp_fto, trail_fto))
features$Away_FTO <- with(regSeason, ifelse(at == "@", trail_fto, trail_opp_fto))
features$Home_Pass_Eff <- with(regSeason, ifelse(at == "@", trail_opp_pass_eff, trail_pass_eff))
features$Away_Pass_Eff <- with(regSeason, ifelse(at == "@", trail_pass_eff, trail_opp_pass_eff))
features$Home_Pass_Def <- with(regSeason, ifelse(at == "@", trail_opp_pass_def, trail_pass_def))
features$Away_Pass_Def <- with(regSeason, ifelse(at == "@", trail_pass_def, trail_opp_pass_def))
features$Home_Rush_Eff <- with(regSeason, ifelse(at == '@', trail_opp_rush_eff, trail_rush_eff))
features$Away_Rush_Eff <- with(regSeason, ifelse(at == "@", trail_rush_eff, trail_opp_rush_eff))
features$Home_Rush_Def <- with(regSeason, ifelse(at == "@", trail_opp_rush_def, trail_rush_def))
features$Away_Rush_Def <- with(regSeason, ifelse(at == "@", trail_rush_def, trail_opp_rush_def))
features$Home_Pen_Yds <- with(regSeason, ifelse(at == "@", trail_opp_penYds, trail_penYds))
features$Away_Pen_Yds <- with(regSeason, ifelse(at == "@", trail_penYds, trail_opp_penYds))
features$Home_Pen_Yds_Agg <- with(regSeason, ifelse(at == "@", trail_opp_penYdsAgg, trail_penYdsAgg))
features$Away_Pen_Yds_Agg <- with(regSeason, ifelse(at == "@", trail_penYdsAgg, trail_opp_penYdsAgg))
features$Home_Third_Eff <- with(regSeason, ifelse(at == "@", trail_opp_third_eff, trail_third_eff))
features$Away_Third_Eff <- with(regSeason, ifelse(at == "@", trail_third_eff, trail_opp_third_eff))
features$Home_Third_Def <- with(regSeason, ifelse(at == "@", trail_opp_third_def, trail_third_def))
features$Away_Third_Def <- with(regSeason, ifelse(at == "@", trail_third_def, trail_opp_third_def))
features$Home_Fourth_Eff <- with(regSeason, ifelse(at == "@", trail_opp_fourth_eff, trail_fourth_eff))
features$Away_Fourth_Eff <- with(regSeason, ifelse(at == "@", trail_fourth_eff, trail_opp_fourth_eff))
features$Home_Fourth_Def <- with(regSeason, ifelse(at == "@", trail_opp_fourth_def, trail_fourth_def))
features$Away_Fourth_Def <- with(regSeason, ifelse(at == "@", trail_fourth_def, trail_opp_fourth_def))
#features$Lost_Last_As_Fav <- regSeason$lostLastAsFav
#features$Won_Last_As_Dog <- regSeason$wonLastAsDog

