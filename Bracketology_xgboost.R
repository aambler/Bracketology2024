library(xgboost)
library(caret)
library(tidyverse)
library(rvest)
library(gt)
library(webshot2)
library(toRvik)
library(curl)
library(cbbdata)
library(cbbplotR)
library(janitor)


setwd("~/Data/Data042323/Data/Bracketology")

cbbdata::cbd_login("aeambler", "Basketb@ll2024#")

# Modeling ######################
{
#### Preparation ###################

  
#data <- read.csv("C:/Users/austin/Documents/Data/Data042323/Data/Bracketology/NET 19,22,23 csv.csv") %>% 
 # select(-Team, -Record, -Conference, -ACC, -Big10, -Big12, -SEC, -Pac12, -ConfUnder80, -Conf80120,
  #       -RankAvg, -RankNetAvg, -QualityAvg, -QualityNetAvg, -BPISagarin, -NETKenpom, -KPISOR,
  #      -Sagarin, -AvgNetofConf, -SOS, -KPI, -Wins, -Losses)


# In 2024, assumed that Mississippi, Syracuse, Memphis, and Indiana would have been
  #  after the 3 seeds in the NIT tournament
  
# with more teams in NIT:
data <- read_csv("NEW_teamdata.csv") %>% 
  select(OverallSeed, NET, Q1Wins, Q1Losses, Q1WinPer, Q2Wins, Q2Losses, Q2WinPer,
         Q3Wins, Q3Losses, Q3WinPer, Q4Wins, Q4Losses, Q4WinPer, KPIRank,
         SOR, BPI, Kenpom)


# Splitting the data into training and testing sets
set.seed(123)
index <- createDataPartition(data$OverallSeed, p = 0.8, list = FALSE)
train_data <- data[index, ]
test_data <- data[-index, ]

# Prepare data for xgboost (xgboost uses a specific data structure called DMatrix)
train_matrix <- xgb.DMatrix(data = as.matrix(train_data %>% select(-OverallSeed)), label = train_data$OverallSeed)
test_matrix <- xgb.DMatrix(data = as.matrix(test_data %>% select(-OverallSeed)), label = test_data$OverallSeed)

# Define parameters (adjust according to your needs)
params <- list(
  booster = "gbtree",
  objective = "reg:squarederror",
  eta = 0.3,
  gamma = 0,
  max_depth = 6,
  min_child_weight = 1,
  subsample = 1,
  colsample_bytree = 1 
)

# Number of rounds for boosting
nrounds <- 500

#### Running the model #######################

# rmse doesn't change after 300th iteration
xgb_model <- xgboost(params = params, data = train_matrix, nrounds = nrounds, verbose = 1)

#xgb.save(xgb_model, "bracketology_model")

xgb_model <- xgb.load("bracketology_model")

#### Tune the Parameters ########################

# This is telling me that the best iteration is the 16th nround ???
xgbcv <- xgb.cv(params = params, data = train_matrix, nrounds = nrounds, nfold = 5, 
                 showsd = T, stratified = T, print.every.n = 10, early.stop.round = 20, maximize = F)

# Write a grid search function to tune parameters
GridSearch <- function(paramDF, dtrain) {
  paramList <- lapply(split(paramDF, 1:nrow(paramDF)), as.list)
  bestResults <- tibble()
  pb <- txtProgressBar(style = 3)
  for(i in seq(length(paramList))) {
    rwCV <- xgb.cv(params = paramList[[i]],
                   data = dtrain, 
                   nrounds = 500, 
                   nfold = 10,
                   early_stopping_rounds = 10,
                   verbose = FALSE)
    bestResults <- bestResults %>% 
      bind_rows(rwCV$evaluation_log[rwCV$best_iteration])
    gc() 
    setTxtProgressBar(pb, i/length(paramList))
  }
  close(pb)
  return(bind_cols(paramDF, bestResults) %>% arrange(test_rmse_mean))
}

# Tune eta
paramDF <- tibble(eta = c(0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4))
eta_search <- GridSearch(paramDF, train_matrix)
eta_search # 0.3 and 0.1

# Tune max depth and max leaves together using eta = 0.1
paramDF <- expand.grid(
  max_depth = seq(2, 8, by = 2),
  max_leaves = c(5, 10, 25, 64),
  eta = 0.1)
depth_leaves <- GridSearch(paramDF, train_matrix)
depth_leaves # 2 for max_depth, and no need to set max_leaves - doesn't matter
# the larger the max_depth, the more memory the computer uses and 
#  the more likely the model is to overfit since the model is more complex

# Tune subsample and colsample_bytree together using eta = 0.1, max_depth = 2
paramDF <- expand.grid(
  subsample = seq(0.5, 1, by = 0.1),
  colsample_bytree = seq(0.5, 1, by = 0.1),
  max_depth = 2,
  eta = 0.1)
set.seed(848)
randsubsets <- GridSearch(paramDF, train_matrix)
randsubsets # 1 for subsample and 0.5 for colsample_by_tree
# due to few variables, use 1 for subsample and colsample_by_tree

# Tune min_child_weight
paramDF <- tibble(min_child_weight = seq(1, 10, by = 1))
paramDF <- expand.grid(
  min_child_weight = seq(1, 10, by = 1),
  subsample = 1,
  colsample_by_tree = 1,
  max_depth = 2,
  eta = 0.1)
child_weights <- GridSearch(paramDF, train_matrix)
child_weights # 5 is best, Maddox said don't need this

# Tune gamma
paramDF <- tibble(gamma = seq(0, 10, by = 1))
gammas <- GridSearch(paramDF, train_matrix)
gammas # 0 is best

# Redefine parameters based on tuning
params <- list(
  booster = "gbtree",
  objective = "reg:squarederror",
  eta = 0.1,
  gamma = 6, # make it 0
  max_depth = 17, # 2
  max_leaves = 1023, # leave at default, get rid of
  min_child_weight = 5, # get rid of
  subsample = 1,
  colsample_bytree = 1)

nrounds <- 500

# By like 500th iteration, the rmse isn't changing
#xgb_model <- xgboost(params = params, data = train_matrix, nrounds = nrounds, verbose = 1)
#xgb_model2 <- xgb.train(params = params, data = train_matrix, nrounds = nrounds, verbose =1,
#                      watchlist = list(train = train_matrix, eval = test_matrix),
#                     early_stopping_rounds = 50)

#xgb.save(xgb_model2, "bracketology_model2")



# Do it a different way - doesn't really change much
# Define parameter grid
param_grid <- expand.grid(
  nrounds = c(1000), # Number of boosting rounds
  max_depth = c(11, 57, 101), # Maximum depth of trees
  eta = c(0.1),     # Learning rate
  gamma = c(0.1, 0.5), # Minimum loss reduction required to make a further partition
  colsample_bytree = c(0.8), # Subsample ratio of columns when constructing each tree
  min_child_weight = c(5, 703, 2103), # Minimum sum of instance weight (hessian) needed in a child
  subsample = c(0.8)
)

ctrl <- trainControl(
  method = "cv",           # Cross-validation
  number = 5,              # Number of folds
  verboseIter = TRUE,      # Print progress
  allowParallel = TRUE     # Allow parallel processing
)

xgb_grid <- train(
  x = data %>% select(-OverallSeed),   # Features
  y = data$OverallSeed,   # Target variable
  method = "xgbTree",          # XGBoost
  trControl = ctrl,            # Control parameters
  tuneGrid = param_grid        # Parameter grid
)

xgb_model <- xgb_grid$finalModel
tune <- xgb_grid$bestTune
#xgb.save(xgb_model, "bracketology_model4")
}
# model is my original model, model2 is what Maddox and I tuned together, 
#  model3 is the grid tune, model4 has more NIT teams
xgb_model <- xgb.load("bracketology_model")

# Determine how important each variable is
{
# Gain is the improvement in accuracy brought by a feature to the branches it is on. 
#  The idea is that before adding a new split on a feature X to the branch there were some wrongly classified elements; 
#  after adding the split on this feature, there are two new branches, 
#  and each of these branches is more accurate 
#  (one branch saying if your observation is on this branch then it should be classified as 1, 
#  and the other branch saying the exact opposite).

# Cover is related to the second order derivative (or Hessian) of the loss function with respect to a particular variable; 
#  thus, a large value indicates a variable has a large potential impact on the loss function and so is important.

# Frequency is a simpler way to measure the Gain. 
#  It just counts the number of times a feature is used in all generated trees. 
#  You should not use it (unless you know why you want to use it).
importance <- xgb.importance(feature_names = colnames(train_matrix), model = xgb_model)

# create a bar plot of the "gain" variable in the importance data frame

# SOR, KPI, and Net are the most important features - SOR and KPI are the results based metrics,
#  seems like the committee cares about the results more than the predictive metrics
xgb.plot.importance(importance_matrix = importance)
}

# # tidymodels XGboost
{
 # library(modelStudio)
  #library(tidymodels)
# fit_xgboost <- boost_tree(learn_rate = 0.1) %>% 
#   set_mode("regression") %>% 
#   set_engine("xgboost") %>% 
#   fit(OverallSeed ~ ., data = data)
# 
# # Create an Explainer
# explainer <- DALEX::explain(
#   model = fit_xgboost,
#   data = data,
#   y = data$OverallSeed,
#   label = "XGBoost"
# )
# 
# modelStudio::modelStudio(explainer)
}

# Tree and Evaulation
{
# Plotting the first tree (modify the index if you want a different tree)

# # can only plot one tree, can't combine them all
# library(DiagrammeR)
# xgb.plot.tree(xgb_model, feature_names = colnames(train_data %>% select(-OverallSeed)), trees = 49)
# 
# 
# # Make predictions and evaluate
# predictions <- predict(xgb_model, newdata = test_matrix)
# 
# # Evaluate the model
# 
# # getting around 4 to 5 - doesn't matter how many iterations i run
# postResample(pred = predictions, obs = test_data$OverallSeed)
# 
# # Plotting actual vs predicted values
# results_df <- data.frame(Actual = test_data$OverallSeed, Predicted = predictions)
# ggplot(results_df, aes(x = Actual, y = Predicted)) +
#   geom_point() +
#   geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "blue") +
#   labs(x = "Actual Overall Seed", y = "Predicted Overall Seed", title = "XGBoost: Predicted vs Actual Seed") +
#   theme_minimal()
}

# Future data #################
{
tourney_sheets2024 <- toRvik::bart_tourney_sheets(year = 2024) 

tourney_sheets2024_filter <- tourney_sheets2024 %>% 
  filter(team != "Team") %>% 
  map_df(~ str_remove_all(.x, " F4O")) %>% 
  map_df(~ str_remove_all(.x, " N4O")) %>% 
  mutate_at(vars(net, kpi, sor, bpi, kp), as.numeric) %>% 
  mutate(Q1Wins = as.numeric(str_extract(q1, "\\d+"))) %>% 
  mutate(Q1Losses = as.numeric(str_extract(str_extract(q1, "\\-\\d+"), "\\d+"))) %>% 
  mutate(Q1WinPer = Q1Wins / (Q1Wins + Q1Losses)) %>% 
  mutate(Q2Wins = as.numeric(str_extract(q2, "\\d+")),
         Q2Losses = as.numeric(str_extract(str_extract(q2, "\\-\\d+"), "\\d+")),
         Q2WinPer = Q2Wins / (Q2Wins + Q2Losses),
         Q3Wins = as.numeric(str_extract(q3, "\\d+")),
         Q3Losses = as.numeric(str_extract(str_extract(q3, "\\-\\d+"), "\\d+")),
         Q3WinPer = Q3Wins / (Q3Wins + Q3Losses),
         Q4Wins = as.numeric(str_extract(q4, "\\d+")),
         Q4Losses = as.numeric(str_extract(str_extract(q4, "\\-\\d+"), "\\d+")),
         Q4WinPer = Q4Wins / (Q4Wins + Q4Losses)) %>% 
  rename("NET" = net, "KPIRank" = kpi, "SOR" = sor, "BPI" = bpi, "Kenpom" = kp) %>% 
  select(-seed, -avg_2, -qual_avg, -q1, -q2, -q1_2, -q3, -q4, -res_avg)

tourney_sheets2024_filter$Q1WinPer[is.na(tourney_sheets2024_filter$Q1WinPer)] <- 0
tourney_sheets2024_filter$Q2WinPer[is.na(tourney_sheets2024_filter$Q2WinPer)] <- 0
tourney_sheets2024_filter$Q3WinPer[is.na(tourney_sheets2024_filter$Q3WinPer)] <- 0
tourney_sheets2024_filter$Q4WinPer[is.na(tourney_sheets2024_filter$Q4WinPer)] <- 0

# kenpom_page <- "https://kenpom.com/"
# kenpom_xml <- read_html(kenpom_page)
# kenpom <- kenpom_xml %>% html_nodes("#data-area") %>% html_table(header = TRUE) %>% pluck(1) %>% 
#   row_to_names(row_number = 1) %>% 
#   clean_names() %>% 
#   select(rk, team) %>% 
#   map_df(~ str_replace_all(.x, "N\\.C\\. State", "North Carolina St.")) %>% 
#   mutate_at(vars(rk), as.numeric)
# 
# tourney_sheets2024_filter <- tourney_sheets2024_filter %>% 
#   left_join(kenpom, by = c("team"))

tourney_sheets2024_filter2 <- tourney_sheets2024_filter %>% 
  select(NET, Q1Wins, Q1Losses, Q1WinPer, Q2Wins, Q2Losses, Q2WinPer, Q3Wins, Q3Losses, Q3WinPer,
         Q4Wins, Q4Losses, Q4WinPer, KPIRank, SOR, BPI, Kenpom)







new_data_dmatrix <- xgb.DMatrix(data = as.matrix(tourney_sheets2024_filter2))

xgboost_predictions <- predict(xgb_model, new_data_dmatrix)

# xgboost_predictions

# Pretty good predictions for 12/7 and the model only had 500 rounds
pred_table_xgboost <- as.data.frame(cbind(tourney_sheets2024_filter$team, as.numeric(xgboost_predictions))) %>% 
  arrange(xgboost_predictions)
pred_table_xgboost



#pred_table_xgboost[8,1] <- "Marquette"
#pred_table_xgboost[10,1] <- "Illinois"
}
# Manipulate ######################
{
#### Add conferences
teams <- toRvik::bart_ratings() %>% 
  select(team, conf) %>% 
  mutate(conf = ifelse(team == "Houston", "B12", conf),
         conf = ifelse(team == "UCF", "B12", conf),
         conf = ifelse(team == "Cincinnati", "B12", conf),
         conf = ifelse(team == "BYU", "B12", conf),
         conf = ifelse(team == "Florida Atlantic", "Amer", conf),
         conf = ifelse(team == "Charlotte", "Amer", conf),
         conf = ifelse(team == "North Texas", "Amer", conf),
         conf = ifelse(team == "Rice", "Amer", conf),
         conf = ifelse(team == "UAB", "Amer", conf),
         conf = ifelse(team == "UTSA", "Amer", conf),
         conf = ifelse(team == "New Mexico St.", "CUSA", conf),
         conf = ifelse(team == "Liberty", "CUSA", conf),
         conf = ifelse(team == "Jacksonville St.", "CUSA", conf),
         conf = ifelse(team == "Sam Houston", "CUSA", conf),
         conf = ifelse(team == "Campbell", "CAA", conf),
         conf = ifelse(team == "Western Illinois", "OVC", conf))

pred_table_xgboost <- pred_table_xgboost %>% 
  left_join(teams, by = c("V1" = "team")) 


pred_table_xgboost <- pred_table_xgboost %>% 
  mutate(conf = ifelse(V1 == "Le Moyne", "NEC", conf)) %>% 
  left_join(tourney_sheets2024_filter, by = c("V1" = "team")) %>% 
  select("Team" = V1, "Rank" = V2, conf, NET)

# get rid of ineligible teams
ineligible_teams <- c("Bellarmine", "Le Moyne",
                      "Lindenwood", "Queens",
                      "St. Thomas", "Southern Indiana",
                      "Stonehill", "Texas A&M Commerce",
                      "Tarleton St.", "UC San Diego",
                      "Utah Tech")

elim_teams <- c("Central Arkansas", "Florida Gulf Coast", "Kennesaw St.", 
                "Eastern Kentucky", "Lipscomb", "North Florida", "Jacksonville", 
                "North Alabama", "Austin Peay", # ASUN
                "Army", "Loyola MD", "Holy Cross", "Lafayette", "Navy", "American", "Bucknell", 
                "Boston University", "Lehigh", # Patriot
                "St. Francis PA", "LIU Brooklyn", "Sacred Heart", "Fairleigh Dickinson",
                "Le Moyne", "Central Connecticut", "Merrimack", # NEC
                "Louisiana Monroe", "Old Dominion", "South Alabama", "Coastal Carolina",
                "Southern Miss", "Georgia St.", "Georgia Southern", "Louisiana Lafayette", 
                "Troy", "Marshall", "Appalachian St.", "Texas St.", "Arkansas St.", # Sun Belt
                "IUPUI", "Robert Morris", "Detroit", "Fort Wayne", "Youngstown St.", "Wright St.", 
                "Green Bay", "Cleveland St.", "Northern Kentucky", "Milwaukee", # Horizon
                "Eastern Illinois", "Tennessee St.", "Western Illinois", "Tennessee Martin",
                "Tennessee Tech", "Southeast Missouri St.", "Little Rock", #OVC
                "Murray St.", "Valparaiso", "Illinois St.", "Southern Illinois", 
                "SIU Edwardsville", "Missouri St.", "Belmont", "Evansville", "Illinois Chicago", 
                "Northern Iowa", "Bradley", # MVC
                "USC Upstate", "Radford", "Winthrop", "Charleston Southern", 
                "Presbyterian", "High Point", "Gardner Webb", "William & Mary", "UNC Asheville", # Big South
                "North Carolina A&T", "Elon", "Campbell", "Northeastern", "Hampton", 
                "Monmouth", "UNC Wilmington", "Drexel", "Delaware", "Towson", "Hofstra", "Stony Brook", # CAA
                "The Citadel", "VMI", "Mercer", "UNC Greensboro", "Wofford", "Western Carolina", 
                "Furman", "Chattanooga", "East Tennessee St.", # SoCon
                "Houston Christian", "Incarnate Word",  "Southeastern Louisiana", "Northwestern St.", 
                "New Orleans", "Texas A&M Commerce", "Lamar", "Texas A&M Corpus Chris", "Nicholls St.", # Southland
                "Pacific", "Pepperdine", "Loyola Marymount", "San Diego", "Portland", 
                "Santa Clara", "San Francisco", # WCC
                "South Dakota", "Oral Roberts", "UMKC", "North Dakota St.", "North Dakota", 
                "Nebraska Omaha", "Denver", # Summit
                "Chicago St." , # Ind
                "Maine", "Albany", "Binghamton", "UMBC", "New Hampshire", "Bryant", "UMass Lowell", # American East
                "Penn", "Harvard", "Columbia", "Dartmouth", "Princeton", "Cornell", "Brown", # Ivy
                "Idaho", "Northern Arizona", "Eastern Washington", "Northern Colorado", 
                "Weber St.", "Portland St.", "Sacramento St.", "Idaho St.", "Montana", # Big Sky
                "Davidson", "George Washington", "Rhode Island", "George Mason", "Fordham", 
                "La Salle", "Saint Louis", "Richmond", "Massachusetts", "Loyola Chicago", 
                "Saint Joseph's", "St. Bonaventure", "VCU", # A10
                "Oklahoma St.", "West Virginia", "UCF", "Cincinnati", "Kansas St.", # Big 12
                "Georgia Tech", "Louisville", "Miami FL", "Virginia Tech", "Notre Dame", 
                "Syracuse", "Florida St.", "Boston College", "Wake Forest", # ACC
                "Mount St. Mary's", "Manhattan", "Siena", "Canisius", "Iona", "Niagara", 
                "Rider", "Quinnipiac", "Marist", "Fairfield", # MAAC
                "Jacksonville St.", "FIU", "Louisiana Tech", "Liberty", "New Mexico St.", "Middle Tennessee", "UTEP", # CUSA
                "Rice", "UTSA", "Tulsa", "Memphis", "Tulane", "SMU", "East Carolina", 
                "Wichita St.", "North Texas", "Charlotte", "South Florida", "Temple", # AAC
                "Wyoming", "San Jose St.", "Air Force", "Fresno St.", "UNLV", # MWC
                "Alcorn St.", "Alabama St.", "Jackson St.", "Southern", "Bethune Cookman", 
                "Alabama A&M", "Texas Southern", # SWAC
                "Washington", "Oregon St.", "California", "Arizona St.", "USC", 
                "UCLA", "Stanford", "Utah", # Pac 12
                "Butler", "Georgetown", "DePaul", "Xavier", "Villanova", # Big East
                "Rutgers", "Michigan", "Minnesota", "Maryland", "Iowa", "Penn St.", 
                "Ohio St.", "Indiana", # Big 10
                "Vanderbilt", "Missouri", "LSU", "Arkansas", "Mississippi", "Georgia", # SEC
                "Coppin St.", "Maryland Eastern Shore", "Morgan St.", "South Carolina St.",
                "Norfolk St.", "North Carolina Central", "Delaware St.", # MEAC
                "Cal St. Bakersfield", "UC Santa Barbara", "Cal St. Northridge", 
                "UC Riverside", "UC Irvine", "Hawaii", "UC Davis", # Big West
                "Utah Valley", "Abilene Christian", "Cal Baptist", "Stephen F. Austin", 
                "Sam Houston St.", "Seattle", "Tarleton St.", "UT Arlington", # WAC
                "Buffalo", "Ball St.", "Toledo", "Central Michigan", "Miami OH", 
                "Western Michigan", "Bowling Green", "Ohio", "Kent St.", # MAC
                "Eastern Michigan", "Northern Illinois", "Prairie View A&M", "Southern Utah",
                "NJIT", "Florida A&M", 
                "UT Rio Grande Valley",
                "Cal Poly", "Mississippi Valley St.", "Cal St. Fullerton",
                "Arkansas Pine Bluff",
                "St. John's", "Pittsburgh", "Seton Hall", "Oklahoma", "Providence", "Indiana St.")

pred_table_xgboost <- pred_table_xgboost %>% 
  filter(!(Team %in% ineligible_teams), !(Team %in% elim_teams))


pred_table_xgboost$Rank <- as.numeric(pred_table_xgboost$Rank)

# Right now there are 9 confs with multi bids (WCC has gonzaga as autobid), so need 23 spots for other auto-bids
#  22 teams take up 12-16 seeds, best auto-bid is the last 11 seed right now, but don't have to play in First Four
#  First Four is between 41-45 right now
# switching WCC to A10
# adding WCC back into Autobid, first four will now be 42-46

pred_multi_bid_conf <- pred_table_xgboost %>% 
  filter(conf %in% c("ACC", "B12", "B10", "P12", "SEC", "BE",
                     "Amer", "A10", "MWC", "WCC", "MVC"),
         NET < 75, Team != "Drake", Team != "Oregon") %>% 
  mutate(overall_rank = rank(Rank)) %>% 
  mutate(seed = case_when(overall_rank == 41 | overall_rank == 42 ~ 10,
                          overall_rank <= 40 ~ ceiling(overall_rank / 4),
                          #overall_rank == 45 ~ 11,
                          #overall_rank == 46 ~ 11,
                          TRUE ~ 0)) %>% 
  filter(seed != 0) %>% 
  select(Team, overall_rank, seed, conf)

oregon_ncstate <- pred_table_xgboost %>% 
  filter(Team == "North Carolina St." | Team == "Oregon") %>% 
  mutate(overall_rank = rank(Rank)) %>% 
  mutate(seed = 11) %>% 
  mutate(overall_rank = overall_rank + 42) %>% 
  select(Team, overall_rank, seed, conf)
  
  


conf_auto_bids <- pred_table_xgboost %>% 
  select(Team, NET, conf) %>% 
  filter(!(conf %in% c("B12", "B10", "SEC", "BE", "ACC", "P12",
                      "MWC", "WCC", "ind")), 
         Team != "Indiana St.", Team != "Dayton", Team != "Florida Atlantic",
         Team != "Arizona", Team != "Colorado", Team != "Washington St.",
         Team != "North Carolina", Team != "Duke", Team != "Clemson", Team != "Virginia",
         Team != "Pittsburgh", Team != "Wake Forest") %>% 
  group_by(conf) %>% 
  mutate(conf_rank = rank(NET)) %>% 
  filter(conf_rank == 1) %>% 
  ungroup() %>% 
  mutate(overall_rank = rank(NET)) %>% 
  mutate(seed = case_when(overall_rank <= 2 ~ 11,
                          #overall_rank <= 21 ~ ceiling((overall_rank -1) / 4 + 11),
                          #overall_rank <= 20 ~ floor((overall_rank - 1) / 4 + 12),
                          #overall_rank >= 1 & overall_rank <= 3 ~ 12,
                          #overall_rank >= 4 & overall_rank <= 7 ~ 13,
                          #overall_rank >= 8 & overall_rank <= 11 ~ 14,
                          #overall_rank >= 12 & overall_rank <= 15 ~ 15,
                          overall_rank >= 3 & overall_rank <= 6 ~ 12,
                          overall_rank >= 7 & overall_rank <= 10 ~ 13,
                          overall_rank >= 11 & overall_rank <= 14 ~ 14,
                          overall_rank >= 15 & overall_rank <= 18 ~ 15,
                          TRUE ~ 16)) %>% 
  mutate(overall_rank = overall_rank + 44) %>% 
  select(Team, overall_rank, seed, conf)

all_tourney_teams <- rbind(pred_multi_bid_conf, oregon_ncstate, conf_auto_bids) %>% 
  arrange(overall_rank) %>% 
  mutate(first_four = ifelse(overall_rank %in% c(39,40,41,42,65,66,67,68), 1, 0)) %>% 
  map_df(~ str_replace(.x, "St\\.", "State")) %>% 
  map_df(~ str_replace(.x, "State Thomas", "St. Thomas")) %>% 
  map_df(~ str_replace_all(.x, "State John's", "St. John's"))

# count how many in each conference
all_tourney_teams %>% 
  group_by(conf) %>% 
  count() %>% 
  arrange(desc(n)) %>% 
  head(8) %>% 
  ggplot(aes(x = reorder(conf, -n), n)) +
  geom_col(fill = "orange", color = "orange") +
  geom_text(aes(label = n), vjust = -.3, color = "black") +
  theme_minimal() +
  theme(
    plot.title = element_text(face = 'bold', size = 14, hjust = 0.5),
    plot.title.position = 'plot',
    axis.text.x = cbbplotR::element_cbb_conferences(size = 0.6),
    axis.text.y = element_blank(),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank()) +
  labs(title = "Number of Bids by Conference",
       x = "",
       y = "")
}

# Location and Distance #############################
{
geography <- read_csv("College Geography Data.csv") %>% 
  select("college" = INSTNM, "latitude" = LATITUDE, "longitude" = LONGITUD) %>% 
  map_df(~ str_replace_all(.x, "California State University-Long Beach", "Long Beach St.")) %>% 
  map_df(~ str_remove_all(.x, "The University of ")) %>%
  map_df(~ str_remove_all(.x, "University of ")) %>% 
  map_df(~ str_remove_all(.x, "University")) %>% 
  map_df(~ str_remove_all(.x, "-Main Campus")) %>% 
  map_df(~ str_replace_all(.x, "A & M", "A&M")) %>% 
  map_df(~ str_remove_all(.x, "-Knoxville")) %>% 
  map_df(~ str_replace(.x, "Texas A&M -College Station", "Texas A&M")) %>%
  map_df(~ str_replace(.x, "Brigham Young", "BYU")) %>% 
  map_df(~ str_replace(.x, "Oklahoma\\-Norman Campus", "Oklahoma")) %>% 
  map_df(~ str_remove(.x, "Urbana-Champaign")) %>% 
  map_df(~ str_replace(.x, "Colorado State -Global Campus", "Colorado State")) %>%
  map_df(~ str_remove_all(.x, "-Madison")) %>%
  map_df(~ str_remove_all(.x, "-Columbia")) %>%
  map_df(~ str_remove_all(.x, "at Chapel Hill")) %>% 
  map_df(~ str_replace(.x, "Miami", "Miami FL")) %>%
  map_df(~ str_remove_all(.x, "Boulder")) %>%
  map_df(~ str_replace(.x, "California-Irvine", "UC Irvine")) %>% 
  map_df(~ str_remove_all(.x, "-Reno")) %>%
  map_df(~ str_replace(.x, "Polytechnic Institute and State", "Tech")) %>% 
  map_df(~ str_replace(.x, "Massachusetts-Lowell", "UMass Lowell")) %>% 
  map_df(~ str_replace(.x, "North Carolina at Greensboro", "UNC Greensboro")) %>% 
  map_df(~ str_remove(.x, "Main Campus")) %>% 
  map_df(~ str_replace(.x, "Marist College", "Marist")) %>% 
  map_df(~ str_remove(.x, "and A&M College")) %>% 
  map_df(~ str_replace(.x, "Merrimack College", "Merrimack")) %>% 
  map_df(~ str_remove_all(.x, "-Lincoln")) %>%
  map_df(~ str_remove_all(.x, "at Austin")) %>%
  map_df(~ str_replace_all(.x, "Texas Christian", "TCU")) %>% 
  map_df(~ str_remove_all(.x, "at Stark")) %>% 
  map_df(~ str_remove_all(.x, "at Raleigh")) %>% 
  map_df(~ str_replace_all(.x, "Saint Mary's College of California", "Saint Mary's")) %>% 
  map_df(~ str_replace_all(.x, "Saint Johns", "St. John's")) %>% 
  map_df(~ str_replace_all(.x, "Purdue  Fort Wayne", "Fort Wayne")) %>% 
  map_df(~ str_replace_all(.x, "North Carolina Wilmington", "UNC Wilmington")) %>% 
  map_df(~ str_replace_all(.x, "Virginia Commonwealth", "VCU")) %>% 
  map_df(~ str_replace_all(.x, "Alabama at Birmingham", "UAB")) %>% 
  map_df(~ str_replace_all(.x, "California-Davis", "UC Davis")) %>% 
  map_df(~ str_replace_all(.x, "Wagner College", "Wagner")) %>%
  map_df(~ str_squish(.x))


all_tourney_teams_loc <- all_tourney_teams %>% 
  left_join(geography, by = c("Team" = "college")) %>% 
  map_df(~ str_replace_all(.x, "State", "St.")) %>% 
  mutate(latitude = ifelse(Team == "Long Beach St.", 33.782818, latitude),
         longitude = ifelse(Team == "Long Beach St.", -118.11204, longitude))

all_tourney_teams_loc$latitude <- as.numeric(all_tourney_teams_loc$latitude)
all_tourney_teams_loc$longitude <- as.numeric(all_tourney_teams_loc$longitude)
all_tourney_teams_loc$overall_rank <- as.numeric(all_tourney_teams_loc$overall_rank)
all_tourney_teams_loc$seed <- as.numeric(all_tourney_teams_loc$seed)

library(REdaS)

#=ACOS(COS(RADIANS(90-$C$3))*COS(RADIANS(90-E3))+
#SIN(RADIANS(90-$C$3))*SIN(RADIANS(90-E3))*COS(RADIANS($D$3-F3)))*3958.8

# LA is 34.055 by -118.244
# Detroit is 42.3314 by - 83.046
# Dallas is 32.777 by -96.809
# Boston is 42.36 by -71.059

# Salt Lake City is 40.761 by -111.891
# Spokane is 47.658 by -117.424
# Memphis is 35.150 by -90.05
# Indy is 39.769 by -86.158
# Pitt is 40.441 by -79.996
# Omaha is 41.257 by -95.935
# Brooklyn is 40.678 by -73.944
# Charlotte is 35.227 by -80.843
all_tourney_teams_loc <- all_tourney_teams_loc %>% 
  mutate(dist_la = acos(cos(deg2rad(90 - latitude)) * cos(deg2rad(90 - 34.055)) +
                          sin(deg2rad(90 - latitude)) * sin(deg2rad(90 - 34.055)) *
                          cos(deg2rad(longitude + 118.244))) * 3958.8,
         dist_detroit = acos(cos(deg2rad(90 - latitude)) * cos(deg2rad(90 - 42.3314)) +
                               sin(deg2rad(90 - latitude)) * sin(deg2rad(90 - 42.3314)) *
                               cos(deg2rad(longitude + 83.046))) * 3958.8,
         dist_dallas = acos(cos(deg2rad(90 - latitude)) * cos(deg2rad(90 - 32.777)) +
                              sin(deg2rad(90 - latitude)) * sin(deg2rad(90 - 32.777)) *
                              cos(deg2rad(longitude + 96.809))) * 3958.8,
         dist_boston = acos(cos(deg2rad(90 - latitude)) * cos(deg2rad(90 - 42.36)) +
                              sin(deg2rad(90 - latitude)) * sin(deg2rad(90 - 42.36)) *
                              cos(deg2rad(longitude + 71.059))) * 3958.8,
         fdist_saltlake = acos(cos(deg2rad(90 - latitude)) * cos(deg2rad(90 - 40.761)) +
                                sin(deg2rad(90 - latitude)) * sin(deg2rad(90 - 40.671)) *
                                cos(deg2rad(longitude + 111.891))) * 3958.8, # doesn't like BYU, Utah, Utah State
         fdist_spokane = acos(cos(deg2rad(90 - latitude)) * cos(deg2rad(90 - 47.658)) +
                               sin(deg2rad(90 - latitude)) * sin(deg2rad(90 - 47.658)) *
                               cos(deg2rad(longitude + 117.424))) * 3958.8,
         fdist_memphis = acos(cos(deg2rad(90 - latitude)) * cos(deg2rad(90 - 35.15)) +
                               sin(deg2rad(90 - latitude)) * sin(deg2rad(90 - 35.15)) *
                               cos(deg2rad(longitude + 90.05))) * 3958.8,
         fdist_indy = acos(cos(deg2rad(90 - latitude)) * cos(deg2rad(90 - 39.769)) +
                            sin(deg2rad(90 - latitude)) * sin(deg2rad(90 - 39.769)) *
                            cos(deg2rad(longitude + 86.158))) * 3958.8,
         fdist_pitt = acos(cos(deg2rad(90 - latitude)) * cos(deg2rad(90 - 40.441)) +
                            sin(deg2rad(90 - latitude)) * sin(deg2rad(90 - 40.441)) *
                            cos(deg2rad(longitude + 79.996))) * 3958.8,
         fdist_omaha = acos(cos(deg2rad(90 - latitude)) * cos(deg2rad(90 - 41.257)) +
                             sin(deg2rad(90 - latitude)) * sin(deg2rad(90 - 41.257)) *
                             cos(deg2rad(longitude + 95.935))) * 3958.8,
         fdist_brooklyn = acos(cos(deg2rad(90 - latitude)) * cos(deg2rad(90 - 40.678)) +
                                sin(deg2rad(90 - latitude)) * sin(deg2rad(90 - 40.678)) *
                                cos(deg2rad(longitude + 73.994))) * 3958.8,
         fdist_charlotte = acos(cos(deg2rad(90 - latitude)) * cos(deg2rad(90 - 35.227)) +
                                 sin(deg2rad(90 - latitude)) * sin(deg2rad(90 - 35.227)) *
                                 cos(deg2rad(longitude + 80.843))) * 3958.8,
         region_pref1 = case_when(
           dist_la < dist_boston & dist_la < dist_dallas & dist_la < dist_detroit ~ "LA",
           dist_boston < dist_la & dist_boston < dist_dallas & dist_boston < dist_detroit ~ "Boston",
           dist_detroit < dist_la & dist_detroit < dist_boston & dist_detroit < dist_dallas ~ "Detroit",
           TRUE ~ "Dallas"
         ),
         region_pref2 = case_when(
           dist_la < dist_boston & dist_la < dist_dallas & dist_la > dist_detroit ~ "LA",
           dist_la < dist_boston & dist_la > dist_dallas & dist_la < dist_detroit ~ "LA",
           dist_la > dist_boston & dist_la < dist_dallas & dist_la < dist_detroit ~ "LA",
           dist_boston < dist_la & dist_boston < dist_dallas & dist_boston > dist_detroit ~ "Boston",
           dist_boston < dist_la & dist_boston > dist_dallas & dist_boston < dist_detroit ~ "Boston",
           dist_boston > dist_la & dist_boston < dist_dallas & dist_boston < dist_detroit ~ "Boston",
           dist_detroit < dist_la & dist_detroit < dist_boston & dist_detroit > dist_dallas ~ "Detroit",
           dist_detroit < dist_la & dist_detroit > dist_boston & dist_detroit < dist_dallas ~ "Detroit",
           dist_detroit > dist_la & dist_detroit < dist_boston & dist_detroit < dist_dallas ~ "Detroit",
           TRUE ~ "Dallas"
         ),
         region_pref3 = case_when(
           dist_la < dist_boston & dist_la > dist_dallas & dist_la > dist_detroit ~ "LA",
           dist_la > dist_boston & dist_la > dist_dallas & dist_la < dist_detroit ~ "LA",
           dist_la > dist_boston & dist_la < dist_dallas & dist_la > dist_detroit ~ "LA",
           dist_boston < dist_la & dist_boston > dist_dallas & dist_boston > dist_detroit ~ "Boston",
           dist_boston > dist_la & dist_boston > dist_dallas & dist_boston < dist_detroit ~ "Boston",
           dist_boston > dist_la & dist_boston < dist_dallas & dist_boston > dist_detroit ~ "Boston",
           dist_detroit < dist_la & dist_detroit > dist_boston & dist_detroit > dist_dallas ~ "Detroit",
           dist_detroit > dist_la & dist_detroit > dist_boston & dist_detroit < dist_dallas ~ "Detroit",
           dist_detroit > dist_la & dist_detroit < dist_boston & dist_detroit > dist_dallas ~ "Detroit",
           TRUE ~ "Dallas"
         ),
         region_pref4 = case_when(
           dist_la > dist_boston & dist_la > dist_dallas & dist_la > dist_detroit ~ "LA",
           dist_boston > dist_la & dist_boston > dist_dallas & dist_boston > dist_detroit ~ "Boston",
           dist_detroit > dist_la & dist_detroit > dist_boston & dist_detroit > dist_dallas ~ "Detroit",
           TRUE ~ "Dallas"
         ))

# Problems with Salt Lake City and Utah teams, so fix it
all_tourney_teams_loc[is.na(all_tourney_teams_loc)] <- 2


all_tourney_teams_loc <- all_tourney_teams_loc %>% 
  mutate(fdist_saltlake = ifelse(Team == "Utah", 3000, fdist_saltlake),
         fdist_omaha = ifelse(Team == "Creighton", 3000, fdist_omaha),
         fdist_memphis = ifelse(Team == "Memphis", 3000, fdist_memphis),
         fdist_brooklyn = ifelse(Team == "BYU", 3000, fdist_brooklyn),
         fdist_indy = ifelse(Team == "BYU", 3000, fdist_indy),
         fdist_spokane = ifelse(Team == "BYU", 3000, fdist_spokane),
         fdist_memphis = ifelse(Team == "Memphis", 3000, fdist_memphis))
}

# Assign the top 16 teams to a region ######################
{
top16 <- all_tourney_teams_loc %>% 
  filter(overall_rank <= 16) %>% 
  mutate(region = "NY")

top16 <- top16 %>% 
  mutate(region = case_when(
    overall_rank == 1 ~ region_pref1))

top16 <- top16 %>% 
  mutate(region = case_when(
    overall_rank == 2 & lag(region) == region_pref1 ~ region_pref2,
    overall_rank == 2 ~ region_pref1,
    TRUE ~ region))

top16 <- top16 %>% 
  mutate(region = case_when(
    overall_rank == 3 & lag(region) == region_pref1 & lag(region, n = 2) == region_pref2 ~ region_pref3,
    overall_rank == 3 & lag(region) == region_pref2 & lag(region, n = 2) == region_pref1 ~ region_pref3,
    overall_rank == 3 & lag(region, n = 2) == region_pref1 & lag(region) != region_pref2 ~ region_pref2,
    overall_rank == 3 & lag(region) == region_pref1 & lag(region, n = 2) != region_pref2 ~ region_pref2,
    overall_rank == 3 ~ region_pref1,
    TRUE ~ region
  ))

options <- data.frame(team = c("Detroit", "Dallas", "Boston", "LA"))

seed3 <- top16[[3, "region"]]
seed2 <- top16[[2, "region"]]
seed1 <- top16[[1, "region"]]

options <- options %>% 
  filter(team != seed3, team != seed2, team != seed1)

top16[4, "region"] = options

top16 <- top16 %>% 
  mutate(region = case_when(
    overall_rank == 5 & region_pref1 == lag(region) & conf != lag(conf) ~ region_pref1,
    overall_rank == 5 & region_pref1 == lag(region, n = 2) & conf != lag(conf, n = 2) ~ region_pref1,
    overall_rank == 5 & region_pref1 == lag(region, n = 3) & conf != lag(conf, n = 3) ~ region_pref1,
    overall_rank == 5 & region_pref1 == lag(region, n = 4) & conf != lag(conf, n = 4) ~ region_pref1,
    overall_rank == 5 ~ region_pref2,
    TRUE ~ region))

region_pref <- top16 %>% 
  filter(overall_rank == 6) %>% 
  select(region_pref1)

region_pref <- as.character(region_pref)

conf_options1 <- top16 %>% 
  filter(region == region_pref) %>% 
  select(conf)

conf_options1 <- as.list(conf_options1)
conf_options1 <- matrix(unlist(conf_options1))

region_pref <- top16 %>% 
  filter(overall_rank == 6) %>% 
  select(region_pref2)

region_pref <- as.character(region_pref)

conf_options2 <- top16 %>% 
  filter(region == region_pref) %>% 
  select(conf)

conf_options2 <- as.list(conf_options2)
conf_options2 <- matrix(unlist(conf_options2))


top16 <- top16 %>% 
  mutate(region = case_when(
    overall_rank == 6 & region_pref1 == lag(region, n = 2) & conf != lag(conf, n = 2) & 
      lag(region) != region_pref1 ~ region_pref1,
    overall_rank == 6 & region_pref1 == lag(region, n = 3) & conf != lag(conf, n = 3) & 
      lag(region) != region_pref1 ~ region_pref1,
    overall_rank == 6 & region_pref1 == lag(region, n = 4) & conf != lag(conf, n = 4) 
    & lag(region) != region_pref1 ~ region_pref1,
    overall_rank == 6 & region_pref1 == lag(region, n = 5) & conf != lag(conf, n = 5) 
    & lag(region) != region_pref1 ~ region_pref1,
    overall_rank == 6 & lag(region) != region_pref2 & !(conf %in% conf_options2) ~ region_pref2,
    overall_rank == 6 ~ region_pref3,
    TRUE ~ region))

# right now 7 is wrong, was giving me duplicate conferences
region_pref <- top16 %>% 
  filter(overall_rank == 7) %>% 
  select(region_pref1)

region_pref <- as.character(region_pref)

conf_options1 <- top16 %>% 
  filter(region == region_pref) %>% 
  select(conf)

conf_options1 <- as.list(conf_options1)
conf_options1 <- matrix(unlist(conf_options1))

region_pref <- top16 %>% 
  filter(overall_rank == 7) %>% 
  select(region_pref2)

region_pref <- as.character(region_pref)

conf_options2 <- top16 %>% 
  filter(region == region_pref) %>% 
  select(conf)

conf_options2 <- as.list(conf_options2)
conf_options2 <- matrix(unlist(conf_options2))

region_pref <- top16 %>% 
  filter(overall_rank == 7) %>% 
  select(region_pref3)

region_pref <- as.character(region_pref)

conf_options3 <- top16 %>% 
  filter(region == region_pref) %>% 
  select(conf)

conf_options3 <- as.list(conf_options3)
conf_options3 <- matrix(unlist(conf_options3))

# same code for 11, think it works, but not 100% sure
top16 <- top16 %>% 
  mutate(region = case_when(
    overall_rank == 7 & region_pref1 != lag(region) & region_pref1 != lag(region, n = 2)
    & !(conf %in% conf_options1) ~ region_pref1,
    overall_rank == 7 & region_pref1 != lag(region) & region_pref1 != lag(region, n = 2)
    & conf %in% conf_options1 
    & region_pref2 != lag(region) & region_pref2 != lag(region, n = 2) ~ region_pref2,
    overall_rank == 7 & region_pref1 == lag(region) & region_pref2 != lag(region, n = 2)
    & !(conf %in% conf_options2) ~ region_pref2,
    overall_rank == 7 & region_pref1 == lag(region, n = 2) & region_pref2 != lag(region)
    & !(conf %in% conf_options2) ~ region_pref2,
    overall_rank == 7 & region_pref1 == lag(region) & region_pref2 == lag(region, n = 2)
    & conf %in% conf_options3 ~ region_pref4,
    overall_rank == 7 & region_pref1 == lag(region, n = 2) & region_pref2 == lag(region)
    & conf %in% conf_options3 ~ region_pref4,
    overall_rank == 7 & conf %in% conf_options1 & region_pref2 == lag(region) 
    & region_pref3 == lag(region, n = 2) ~ region_pref4,
    overall_rank == 7 & conf %in% conf_options1 & region_pref2 == lag(region, n = 2)
    & region_pref3 == lag(region) ~ region_pref4,
    overall_rank == 7 & conf %in% conf_options2 & region_pref2 == lag(region) 
    & region_pref3 == lag(region, n = 2) ~ region_pref4,
    overall_rank == 7 & conf %in% conf_options2 & region_pref2 == lag(region, n = 2)
    & region_pref3 == lag(region) ~ region_pref4,
    overall_rank == 7 & lag(region) != region_pref3 ~ region_pref3,
    overall_rank == 7 ~ region_pref4,
    TRUE ~ region
  ))



options <- data.frame(team = c("Detroit", "Dallas", "Boston", "LA"))

seed3 <- top16[[7, "region"]]
seed2 <- top16[[6, "region"]]
seed1 <- top16[[5, "region"]]

options <- options %>% 
  filter(team != seed3, team != seed2, team != seed1)

top16[8, "region"] = options


region_pref <- top16 %>% 
  filter(overall_rank == 9) %>% 
  select(region_pref1)

region_pref <- as.character(region_pref)

conf_options1 <- top16 %>% 
  filter(region == region_pref) %>% 
  select(conf)

conf_options1 <- as.list(conf_options1)
conf_options1 <- matrix(unlist(conf_options1))

region_pref <- top16 %>% 
  filter(overall_rank == 9) %>% 
  select(region_pref2)

region_pref <- as.character(region_pref)

conf_options2 <- top16 %>% 
  filter(region == region_pref) %>% 
  select(conf)

conf_options2 <- as.list(conf_options2)
conf_options2 <- matrix(unlist(conf_options2))

region_pref <- top16 %>% 
  filter(overall_rank == 9) %>% 
  select(region_pref3)

region_pref <- as.character(region_pref)

conf_options3 <- top16 %>% 
  filter(region == region_pref) %>% 
  select(conf)

conf_options3 <- as.list(conf_options3)
conf_options3 <- matrix(unlist(conf_options3))

top16 <- top16 %>% 
  mutate(region = case_when(
    overall_rank == 9 & conf %in% conf_options1 & conf %in% conf_options2 ~ region_pref3,
    overall_rank == 9 & conf %in% conf_options1 & !(conf %in% conf_options2) ~ region_pref2,
    overall_rank == 9 & !(conf %in% conf_options1) & conf %in% conf_options2 ~ region_pref3,
    overall_rank == 9 ~ region_pref1,
    TRUE ~ region
  ))

region_pref <- top16 %>% 
  filter(overall_rank == 10) %>% 
  select(region_pref1)

region_pref <- as.character(region_pref)

conf_options1 <- top16 %>% 
  filter(region == region_pref) %>% 
  select(conf)

conf_options1 <- as.list(conf_options1)
conf_options1 <- matrix(unlist(conf_options1))

region_pref <- top16 %>% 
  filter(overall_rank == 10) %>% 
  select(region_pref2)

region_pref <- as.character(region_pref)

conf_options2 <- top16 %>% 
  filter(region == region_pref) %>% 
  select(conf)

conf_options2 <- as.list(conf_options2)
conf_options2 <- matrix(unlist(conf_options2))

region_pref <- top16 %>% 
  filter(overall_rank == 10) %>% 
  select(region_pref3)

region_pref <- as.character(region_pref)

conf_options3 <- top16 %>% 
  filter(region == region_pref) %>% 
  select(conf)

conf_options3 <- as.list(conf_options3)
conf_options3 <- matrix(unlist(conf_options3))


top16 <- top16 %>% 
  mutate(region = case_when(
    overall_rank == 10 & !(conf %in% conf_options1) & lag(region) != region_pref1 ~ region_pref1,
    overall_rank == 10 & region_pref2 != lag(region) ~ region_pref2,
    overall_rank == 10 & !(conf %in% conf_options1) ~ region_pref2,
    overall_rank == 10 & !(conf %in% conf_options3) ~ region_pref3,
    overall_rank == 10 ~ region_pref4,
    TRUE ~ region
  ))

region_pref <- top16 %>% 
  filter(overall_rank == 11) %>% 
  select(region_pref1)

region_pref <- as.character(region_pref)

conf_options1 <- top16 %>% 
  filter(region == region_pref) %>% 
  select(conf)

conf_options1 <- as.list(conf_options1)
conf_options1 <- matrix(unlist(conf_options1))

region_pref <- top16 %>% 
  filter(overall_rank == 11) %>% 
  select(region_pref2)

region_pref <- as.character(region_pref)

conf_options2 <- top16 %>% 
  filter(region == region_pref) %>% 
  select(conf)

conf_options2 <- as.list(conf_options2)
conf_options2 <- matrix(unlist(conf_options2))

region_pref <- top16 %>% 
  filter(overall_rank == 11) %>% 
  select(region_pref3)

region_pref <- as.character(region_pref)

conf_options3 <- top16 %>% 
  filter(region == region_pref) %>% 
  select(conf)

conf_options3 <- as.list(conf_options3)
conf_options3 <- matrix(unlist(conf_options3))

# no clue if this is right, got to keep trying examples to see if it works
top16 <- top16 %>% 
  mutate(region = case_when(
    overall_rank == 11 & region_pref1 != lag(region) & region_pref1 != lag(region, n = 2)
     & !(conf %in% conf_options1) ~ region_pref1,
    overall_rank == 11 & region_pref1 != lag(region) & region_pref1 == lag(region, n = 2)
     & conf %in% conf_options1 
     & region_pref2 != lag(region) & region_pref2 != lag(region, n = 2) ~ region_pref2,
    overall_rank == 11 & region_pref1 == lag(region) & region_pref2 != lag(region, n = 2)
     & !(conf %in% conf_options1) ~ region_pref2,
    overall_rank == 11 & region_pref1 == lag(region, n = 2) & region_pref2 != lag(region)
     & !(conf %in% conf_options2) ~ region_pref2,
    overall_rank == 11 & region_pref1 == lag(region) & region_pref2 == lag(region, n = 2)
     & conf %in% conf_options3 ~ region_pref4,
    overall_rank == 11 & region_pref1 == lag(region, n = 2) & region_pref2 == lag(region)
     & conf %in% conf_options3 ~ region_pref4,
    overall_rank == 11 & conf %in% conf_options1 & region_pref2 == lag(region) 
     & region_pref3 == lag(region, n = 2) ~ region_pref4,
    overall_rank == 11 & conf %in% conf_options1 & region_pref2 == lag(region, n = 2)
     & region_pref3 == lag(region) ~ region_pref4,
    overall_rank == 11 & conf %in% conf_options2 & region_pref2 == lag(region) 
    & region_pref3 == lag(region, n = 2) ~ region_pref4,
    overall_rank == 11 & conf %in% conf_options2 & region_pref2 == lag(region, n = 2)
    & region_pref3 == lag(region) ~ region_pref4,
    overall_rank == 11 & conf %in% conf_options2 & region_pref1 == lag(region, n =2) &
      region_pref3 == lag(region) ~ region_pref4,
    overall_rank == 11 & !(region_pref3 == lag(region, n=2)) ~ region_pref3,
    overall_rank == 11 ~ region_pref4,
    TRUE ~ region
  ))

options <- data.frame(team = c("Detroit", "Dallas", "Boston", "LA"))

seed3 <- top16[[11, "region"]]
seed2 <- top16[[10, "region"]]
seed1 <- top16[[9, "region"]]

options <- options %>% 
  filter(team != seed3, team != seed2, team != seed1)

top16[12, "region"] = options



region_pref <- top16 %>% 
  filter(overall_rank == 13) %>% 
  select(region_pref1)

region_pref <- as.character(region_pref)

conf_options1 <- top16 %>% 
  filter(region == region_pref) %>% 
  select(conf)

conf_options1 <- as.list(conf_options1)
conf_options1 <- matrix(unlist(conf_options1))

region_pref <- top16 %>% 
  filter(overall_rank == 13) %>% 
  select(region_pref2)

region_pref <- as.character(region_pref)

conf_options2 <- top16 %>% 
  filter(region == region_pref) %>% 
  select(conf)

conf_options2 <- as.list(conf_options2)
conf_options2 <- matrix(unlist(conf_options2))

region_pref <- top16 %>% 
  filter(overall_rank == 13) %>% 
  select(region_pref3)

region_pref <- as.character(region_pref)

conf_options3 <- top16 %>% 
  filter(region == region_pref) %>% 
  select(conf)

conf_options3 <- as.list(conf_options3)
conf_options3 <- matrix(unlist(conf_options3))

top16 <- top16 %>% 
  mutate(region = case_when(
    overall_rank == 13 & !(conf %in% conf_options1) ~ region_pref1,
    overall_rank == 13 & !(conf %in% conf_options2) ~ region_pref2,
    overall_rank == 13 & !(conf %in% conf_options3) ~ region_pref3,
    overall_rank == 13 ~ region_pref4,
    TRUE ~ region
  ))


region_pref <- top16 %>% 
  filter(overall_rank == 14) %>% 
  select(region_pref1)

region_pref <- as.character(region_pref)

conf_options1 <- top16 %>% 
  filter(region == region_pref) %>% 
  select(conf)

conf_options1 <- as.list(conf_options1)
conf_options1 <- matrix(unlist(conf_options1))

region_pref <- top16 %>% 
  filter(overall_rank == 14) %>% 
  select(region_pref2)

region_pref <- as.character(region_pref)

conf_options2 <- top16 %>% 
  filter(region == region_pref) %>% 
  select(conf)

conf_options2 <- as.list(conf_options2)
conf_options2 <- matrix(unlist(conf_options2))

region_pref <- top16 %>% 
  filter(overall_rank == 14) %>% 
  select(region_pref3)

region_pref <- as.character(region_pref)

conf_options3 <- top16 %>% 
  filter(region == region_pref) %>% 
  select(conf)

conf_options3 <- as.list(conf_options3)
conf_options3 <- matrix(unlist(conf_options3))


top16 <- top16 %>% 
  mutate(region = case_when(
    overall_rank == 14 & !(conf %in% conf_options1) & lag(region) != region_pref1 ~ region_pref1,
    overall_rank == 14 & !(conf %in% conf_options2) & region_pref2 != lag(region) ~ region_pref2,
    overall_rank == 14 & !(conf %in% conf_options3) & region_pref3 != lag(region) ~ region_pref3,
    overall_rank == 14 & (conf %in% conf_options1) & lag(region) == region_pref1 &
      !(conf %in% conf_options2) ~ region_pref2,
    overall_rank == 14 & (conf %in% conf_options1) & lag(region) == region_pref3 ~ region_pref4,
    overall_rank == 14 & conf %in% conf_options1 & conf %in% conf_options2 &
      conf %in% conf_options3 ~ region_pref4,
    overall_rank == 14 ~ region_pref3,
    TRUE ~ region
  ))

# 15
region_pref <- top16 %>% 
  filter(overall_rank == 15) %>% 
  select(region_pref1)

region_pref <- as.character(region_pref)

conf_options1 <- top16 %>% 
  filter(region == region_pref) %>% 
  select(conf)

conf_options1 <- as.list(conf_options1)
conf_options1 <- matrix(unlist(conf_options1))

region_pref <- top16 %>% 
  filter(overall_rank == 15) %>% 
  select(region_pref2)

region_pref <- as.character(region_pref)

conf_options2 <- top16 %>% 
  filter(region == region_pref) %>% 
  select(conf)

conf_options2 <- as.list(conf_options2)
conf_options2 <- matrix(unlist(conf_options2))

region_pref <- top16 %>% 
  filter(overall_rank == 15) %>% 
  select(region_pref3)

region_pref <- as.character(region_pref)

conf_options3 <- top16 %>% 
  filter(region == region_pref) %>% 
  select(conf)

conf_options3 <- as.list(conf_options3)
conf_options3 <- matrix(unlist(conf_options3))

# no clue if this is right, got to keep trying examples to see if it works
top16 <- top16 %>% 
  mutate(region = case_when(
    overall_rank == 15 & region_pref1 != lag(region) & region_pref1 != lag(region, n = 2)
    & !(conf %in% conf_options1) ~ region_pref1,
    overall_rank == 15 & region_pref1 != lag(region) & region_pref1 != lag(region, n = 2)
    & conf %in% conf_options1 
    & region_pref2 != lag(region) & region_pref2 != lag(region, n = 2) ~ region_pref2,
    overall_rank == 15 & region_pref1 == lag(region) & region_pref2 != lag(region, n = 2)
    & !(conf %in% conf_options1) ~ region_pref2,
    overall_rank == 15 & region_pref1 == lag(region, n = 2) & region_pref2 != lag(region)
    & !(conf %in% conf_options1) ~ region_pref2,
    overall_rank == 15 & region_pref2 != lag(region) & region_pref2 != lag(region, n = 2) &
      !(conf %in% conf_options2) ~ region_pref2,
    overall_rank == 15 & region_pref1 == lag(region) & region_pref2 == lag(region, n = 2)
    & conf %in% conf_options3 ~ region_pref4,
    overall_rank == 15 & region_pref1 == lag(region, n = 2) & region_pref2 == lag(region)
    & conf %in% conf_options3 ~ region_pref4,
    overall_rank == 15 & conf %in% conf_options1 & region_pref2 == lag(region) 
    & region_pref3 == lag(region, n = 2) ~ region_pref4,
    overall_rank == 15 & conf %in% conf_options1 & region_pref2 == lag(region, n = 2)
    & region_pref3 == lag(region) ~ region_pref4,
    overall_rank == 15 & conf %in% conf_options2 & region_pref2 == lag(region) 
    & region_pref3 == lag(region, n = 2) ~ region_pref4,
    overall_rank == 15 & conf %in% conf_options2 & region_pref2 == lag(region, n = 2)
    & region_pref3 == lag(region) ~ region_pref4,
    overall_rank == 15 & region_pref3 != lag(region) & !(conf %in% conf_options3) ~ region_pref3,
    overall_rank == 15 ~ region_pref4,
    TRUE ~ region
  ))

options <- data.frame(team = c("Detroit", "Dallas", "Boston", "LA"))

seed3 <- top16[[15, "region"]]
seed2 <- top16[[14, "region"]]
seed1 <- top16[[13, "region"]]

options <- options %>% 
  filter(team != seed3, team != seed2, team != seed1)

top16[16, "region"] = options

top16 <- top16 %>% 
   select(Team, region)
}

#### fix based on if last team in each seed is wrong based on conferences ########
# have to change every time
top16[14,2] <- "Dallas"
top16[16,2] <- "Detroit"


 
all_tourney_teams_loc <- all_tourney_teams_loc %>% 
   left_join(top16, by = "Team")


# Works, but not for conf:::
{
# Top team for each seed
# top16 <- all_tourney_teams_loc %>% 
#   filter(overall_rank <= 16) %>% 
#   mutate(region = case_when(
#     overall_rank == 1 ~ region_pref1,
#     overall_rank == 5 ~ region_pref1,
#     overall_rank == 9 ~ region_pref1,
#     overall_rank == 13 ~ region_pref1,
#     TRUE ~ region_pref1
#   ))
# 
# # 2nd team for each seed
# top16 <- top16 %>% 
#   mutate(region = case_when(
#     overall_rank == 2 & lag(region) == region_pref1 ~ region_pref2,
#     overall_rank == 6 & lag(region) == region_pref1 ~ region_pref2,
#     overall_rank == 10 & lag(region) == region_pref1 ~ region_pref2,
#     overall_rank == 14 & lag(region) == region_pref1 ~ region_pref2,
#     TRUE ~ region_pref1
#   ))
# 
# # 3rd team for each seed - line that is commented out was the problem - should be working now, have to wait and see for other scenarios
# top16 <- top16 %>%
#   mutate(region = case_when(
#     overall_rank == 3 & lag(region) == region_pref1 & lag(region, n = 2) == region_pref2 ~ region_pref3,
#     overall_rank == 3 & lag(region) == region_pref2 & lag(region, n = 2) == region_pref1 ~ region_pref3,
#     overall_rank == 3 & lag(region, n = 2) == region_pref1 & lag(region) != region_pref2 ~ region_pref2,
#     overall_rank == 3 & lag(region) == region_pref1 & lag(region, n = 2) != region_pref2 ~ region_pref2,
#     #overall_rank == 3 & lag(region) == region_pref2 & lag(region, n = 2) != region_pref1 ~ region_pref2,
#     overall_rank == 3 ~ region_pref1,
#     TRUE ~ region),
#     region = case_when(
#       overall_rank == 7 & lag(region) == region_pref1 & lag(region, n = 2) == region_pref2 ~ region_pref3,
#       overall_rank == 7 & lag(region) == region_pref2 & lag(region, n = 2) == region_pref1 ~ region_pref3,
#       overall_rank == 7 & lag(region, n = 2) == region_pref1 & lag(region) != region_pref2 ~ region_pref2,
#       overall_rank == 7 & lag(region) == region_pref1 & lag(region, n = 2) != region_pref2 ~ region_pref2,
#       #overall_rank == 7 & (lag(region) == region_pref2) & (lag(region, n = 2) != region_pref1) ~ region_pref2,
#       overall_rank == 7 ~ region_pref1,
#       TRUE ~ region),
#     region = case_when(
#       overall_rank == 11 & lag(region) == region_pref1 & lag(region, n = 2) == region_pref2 ~ region_pref3,
#       overall_rank == 11 & lag(region) == region_pref2 & lag(region, n = 2) == region_pref1 ~ region_pref3,
#       overall_rank == 11 & lag(region, n = 2) == region_pref1 & lag(region) != region_pref2 ~ region_pref2,
#       overall_rank == 11 & lag(region) == region_pref1 & lag(region, n = 2) != region_pref2 ~ region_pref2,
#      # overall_rank == 11 & lag(region) == region_pref2 & lag(region, n = 2) != region_pref1 ~ region_pref2,
#       overall_rank == 11 ~ region_pref1,
#       TRUE ~ region),
#     region = case_when(
#       overall_rank == 15 & lag(region) == region_pref1 & lag(region, n = 2) == region_pref2 ~ region_pref3,
#       overall_rank == 15 & lag(region) == region_pref2 & lag(region, n = 2) == region_pref1 ~ region_pref3,
#       overall_rank == 15 & lag(region, n = 2) == region_pref1 & lag(region) != region_pref2 ~ region_pref2,
#       overall_rank == 15 & lag(region) == region_pref1 & lag(region, n = 2) != region_pref2 ~ region_pref2,
#       #overall_rank == 15 & lag(region) == region_pref2 & lag(region, n = 2) != region_pref1 ~ region_pref2,
#       overall_rank == 15 ~ region_pref1,
#       TRUE ~ region)
#     )
# 
# # 4th team for each seed
# # figure out the 3 previous regions and remove them from the list, the last one is then the region for the 4th team for each seed
# options <- data.frame(team = c("Detroit", "Dallas", "Boston", "LA"))
# 
# seed3 <- top16[[3, "region"]]
# seed2 <- top16[[2, "region"]]
# seed1 <- top16[[1, "region"]]
# 
# options <- options %>% 
#   filter(team != seed3, team != seed2, team != seed1)
# 
# top16[4, "region"] = options
# 
# # repeat for 8th seed
# options <- data.frame(team = c("Detroit", "Dallas", "Boston", "LA"))
# 
# seed3 <- top16[[7, "region"]]
# seed2 <- top16[[6, "region"]]
# seed1 <- top16[[5, "region"]]
# 
# options <- options %>% 
#   filter(team != seed3, team != seed2, team != seed1)
# 
# top16[8, "region"] = options
# 
# options <- data.frame(team = c("Detroit", "Dallas", "Boston", "LA"))
# 
# seed3 <- top16[[11, "region"]]
# seed2 <- top16[[10, "region"]]
# seed1 <- top16[[9, "region"]]
# 
# options <- options %>% 
#   filter(team != seed3, team != seed2, team != seed1)
# 
# top16[12, "region"] = options
# 
# options <- data.frame(team = c("Detroit", "Dallas", "Boston", "LA"))
# 
# seed3 <- top16[[15, "region"]]
# seed2 <- top16[[14, "region"]]
# seed1 <- top16[[13, "region"]]
# 
# options <- options %>% 
#   filter(team != seed3, team != seed2, team != seed1)
# 
# top16[16, "region"] = options
# 
# top16 <- top16 %>% 
#   select(Team, region)
# 
# all_tourney_teams_loc <- all_tourney_teams_loc %>% 
#   left_join(top16, by = "Team")
}

  
# First Weekend Cities ################################
{
city_columns <- grep("^fdist", colnames(all_tourney_teams_loc), value = TRUE)

# This finds the minimum value of the city_columns,           - right here says negative, without the -, would do maximum
#  if there is a tie, do the fist one, should not be an issue with my problem
all_tourney_teams_loc$first_round_pref1 <- city_columns[max.col(-all_tourney_teams_loc[city_columns], ties.method = "first")]
all_tourney_teams_loc$first_round_pref2 <- city_columns[apply(all_tourney_teams_loc[, city_columns], 1, function(x) order(x)[2])]
all_tourney_teams_loc$first_round_pref3 <- city_columns[apply(all_tourney_teams_loc[, city_columns], 1, function(x) order(x)[3])]
all_tourney_teams_loc$first_round_pref4 <- city_columns[apply(all_tourney_teams_loc[, city_columns], 1, function(x) order(x)[4])]
all_tourney_teams_loc$first_round_pref5 <- city_columns[apply(all_tourney_teams_loc[, city_columns], 1, function(x) order(x)[5])]
all_tourney_teams_loc$first_round_pref6 <- city_columns[apply(all_tourney_teams_loc[, city_columns], 1, function(x) order(x)[6])]
all_tourney_teams_loc$first_round_pref7 <- city_columns[apply(all_tourney_teams_loc[, city_columns], 1, function(x) order(x)[7])]
all_tourney_teams_loc$first_round_pref8 <- city_columns[apply(all_tourney_teams_loc[, city_columns], 1, function(x) order(x)[8])]


all_tourney_teams_loc <- all_tourney_teams_loc %>% 
  map_df(~ str_remove_all(.x, "fdist_"))

# Start assigning 1st round weekends
# There are 16 groups of 4
# 8 locations
# Each location has to be used twice

#### 1-4 Seeds #########################


all_tourney_teams_loc <- all_tourney_teams_loc %>% 
  mutate(saltlake = 0, spokane = 0, memphis = 0, indy = 0, pitt = 0,
         omaha = 0, brooklyn = 0, charlotte = 0)

# 1st and 2nd rank are guaranteed first choice
all_tourney_teams_loc <- all_tourney_teams_loc %>% 
  mutate(first_round_loc = case_when(
    overall_rank == 1 | overall_rank == 2 ~ first_round_pref1))

all_tourney_teams_loc[] <- lapply(all_tourney_teams_loc, function(x) if (is.numeric(x)) replace(x, is.na(x), 0) else replace(x, is.na(x), ""))



get_city <- function(x, df) {
  df <- df %>% 
    mutate(first_round_loc = case_when(
      overall_rank == x & first_round_pref1 == "spokane" & cumsum(spokane) < 2 ~ "spokane",
      overall_rank == x & first_round_pref1 == "saltlake" & cumsum(saltlake) < 2 ~ "saltlake",
      overall_rank == x & first_round_pref1 == "memphis" & cumsum(memphis) < 2 ~ "memphis",
      overall_rank == x & first_round_pref1 == "indy" & cumsum(indy) < 2 ~ "indy",
      overall_rank == x & first_round_pref1 == "pitt" & cumsum(pitt) < 2 ~ "pitt",
      overall_rank == x & first_round_pref1 == "omaha" & cumsum(omaha) < 2 ~ "omaha",
      overall_rank == x & first_round_pref1 == "brooklyn" & cumsum(brooklyn) < 2 ~ "brooklyn",
      overall_rank == x & first_round_pref1 == "charlotte" & cumsum(charlotte) < 2 ~ "charlotte",
      overall_rank == x & first_round_pref2 == "spokane" & cumsum(spokane) < 2 ~ "spokane",
      overall_rank == x & first_round_pref2 == "saltlake" & cumsum(saltlake) < 2 ~ "saltlake",
      overall_rank == x & first_round_pref2 == "memphis" & cumsum(memphis) < 2 ~ "memphis",
      overall_rank == x & first_round_pref2 == "indy" & cumsum(indy) < 2 ~ "indy",
      overall_rank == x & first_round_pref2 == "pitt" & cumsum(pitt) < 2 ~ "pitt",
      overall_rank == x & first_round_pref2 == "omaha" & cumsum(omaha) < 2 ~ "omaha",
      overall_rank == x & first_round_pref2 == "brooklyn" & cumsum(brooklyn) < 2 ~ "brooklyn",
      overall_rank == x & first_round_pref2 == "charlotte" & cumsum(charlotte) < 2 ~ "charlotte",
      overall_rank == x & first_round_pref3 == "spokane" & cumsum(spokane) < 2 ~ "spokane",
      overall_rank == x & first_round_pref3 == "saltlake" & cumsum(saltlake) < 2 ~ "saltlake",
      overall_rank == x & first_round_pref3 == "memphis" & cumsum(memphis) < 2 ~ "memphis",
      overall_rank == x & first_round_pref3 == "indy" & cumsum(indy) < 2 ~ "indy",
      overall_rank == x & first_round_pref3 == "pitt" & cumsum(pitt) < 2 ~ "pitt",
      overall_rank == x & first_round_pref3 == "omaha" & cumsum(omaha) < 2 ~ "omaha",
      overall_rank == x & first_round_pref3 == "brooklyn" & cumsum(brooklyn) < 2 ~ "brooklyn",
      overall_rank == x & first_round_pref3 == "charlotte" & cumsum(charlotte) < 2 ~ "charlotte",
      overall_rank == x & first_round_pref4 == "spokane" & cumsum(spokane) < 2 ~ "spokane",
      overall_rank == x & first_round_pref4 == "saltlake" & cumsum(saltlake) < 2 ~ "saltlake",
      overall_rank == x & first_round_pref4 == "memphis" & cumsum(memphis) < 2 ~ "memphis",
      overall_rank == x & first_round_pref4 == "indy" & cumsum(indy) < 2 ~ "indy",
      overall_rank == x & first_round_pref4 == "pitt" & cumsum(pitt) < 2 ~ "pitt",
      overall_rank == x & first_round_pref4 == "omaha" & cumsum(omaha) < 2 ~ "omaha",
      overall_rank == x & first_round_pref4 == "brooklyn" & cumsum(brooklyn) < 2 ~ "brooklyn",
      overall_rank == x & first_round_pref4 == "charlotte" & cumsum(charlotte) < 2 ~ "charlotte",
      overall_rank == x & first_round_pref5 == "spokane" & cumsum(spokane) < 2 ~ "spokane",
      overall_rank == x & first_round_pref5 == "saltlake" & cumsum(saltlake) < 2 ~ "saltlake",
      overall_rank == x & first_round_pref5 == "memphis" & cumsum(memphis) < 2 ~ "memphis",
      overall_rank == x & first_round_pref5 == "indy" & cumsum(indy) < 2 ~ "indy",
      overall_rank == x & first_round_pref5 == "pitt" & cumsum(pitt) < 2 ~ "pitt",
      overall_rank == x & first_round_pref5 == "omaha" & cumsum(omaha) < 2 ~ "omaha",
      overall_rank == x & first_round_pref5 == "brooklyn" & cumsum(brooklyn) < 2 ~ "brooklyn",
      overall_rank == x & first_round_pref5 == "charlotte" & cumsum(charlotte) < 2 ~ "charlotte",
      overall_rank == x & first_round_pref6 == "spokane" & cumsum(spokane) < 2 ~ "spokane",
      overall_rank == x & first_round_pref6 == "saltlake" & cumsum(saltlake) < 2 ~ "saltlake",
      overall_rank == x & first_round_pref6 == "memphis" & cumsum(memphis) < 2 ~ "memphis",
      overall_rank == x & first_round_pref6 == "indy" & cumsum(indy) < 2 ~ "indy",
      overall_rank == x & first_round_pref6 == "pitt" & cumsum(pitt) < 2 ~ "pitt",
      overall_rank == x & first_round_pref6 == "omaha" & cumsum(omaha) < 2 ~ "omaha",
      overall_rank == x & first_round_pref6 == "brooklyn" & cumsum(brooklyn) < 2 ~ "brooklyn",
      overall_rank == x & first_round_pref6 == "charlotte" & cumsum(charlotte) < 2 ~ "charlotte",
      overall_rank == x & first_round_pref7 == "spokane" & cumsum(spokane) < 2 ~ "spokane",
      overall_rank == x & first_round_pref7 == "saltlake" & cumsum(saltlake) < 2 ~ "saltlake",
      overall_rank == x & first_round_pref7 == "memphis" & cumsum(memphis) < 2 ~ "memphis",
      overall_rank == x & first_round_pref7 == "indy" & cumsum(indy) < 2 ~ "indy",
      overall_rank == x & first_round_pref7 == "pitt" & cumsum(pitt) < 2 ~ "pitt",
      overall_rank == x & first_round_pref7 == "omaha" & cumsum(omaha) < 2 ~ "omaha",
      overall_rank == x & first_round_pref7 == "brooklyn" & cumsum(brooklyn) < 2 ~ "brooklyn",
      overall_rank == x & first_round_pref7 == "charlotte" & cumsum(charlotte) < 2 ~ "charlotte",
      overall_rank == x & first_round_pref8 == "spokane" & cumsum(spokane) < 2 ~ "spokane",
      overall_rank == x & first_round_pref8 == "saltlake" & cumsum(saltlake) < 2 ~ "saltlake",
      overall_rank == x & first_round_pref8 == "memphis" & cumsum(memphis) < 2 ~ "memphis",
      overall_rank == x & first_round_pref8 == "indy" & cumsum(indy) < 2 ~ "indy",
      overall_rank == x & first_round_pref8 == "pitt" & cumsum(pitt) < 2 ~ "pitt",
      overall_rank == x & first_round_pref8 == "omaha" & cumsum(omaha) < 2 ~ "omaha",
      overall_rank == x & first_round_pref8 == "brooklyn" & cumsum(brooklyn) < 2 ~ "brooklyn",
      overall_rank == x & first_round_pref8 == "charlotte" & cumsum(charlotte) < 2 ~ "charlotte",
      TRUE ~ first_round_loc
    ))
  
  df <- df %>% 
    mutate(saltlake = ifelse(first_round_loc == "saltlake", 1, 0),
           spokane = ifelse(first_round_loc == "spokane", 1, 0),
           memphis = ifelse(first_round_loc == "memphis", 1, 0),
           indy = ifelse(first_round_loc == "indy", 1, 0),
           pitt = ifelse(first_round_loc == "pitt", 1, 0),
           omaha = ifelse(first_round_loc == "omaha", 1, 0),
           brooklyn = ifelse(first_round_loc == "brooklyn", 1, 0),
           charlotte = ifelse(first_round_loc == "charlotte", 1, 0)) 
}

for (i in 3:16) {
  all_tourney_teams_loc <- get_city(i, all_tourney_teams_loc)
}


#### 5-16 Seeds #######################
first_round_cities <- function(df, x) {

# Team 17
pref1 <- df %>% 
  filter(overall_rank == x) %>% 
  select(first_round_pref1)

pref1 <- as.character(pref1)

pref2 <- df %>% 
  filter(overall_rank == x) %>% 
  select(first_round_pref2)

pref2 <- as.character(pref2)

pref3 <- df %>% 
  filter(overall_rank == x) %>% 
  select(first_round_pref3)

pref3 <- as.character(pref3)

pref4 <- df %>% 
  filter(overall_rank == x) %>% 
  select(first_round_pref4)

pref4 <- as.character(pref4)

pref5 <- df %>% 
  filter(overall_rank == x) %>% 
  select(first_round_pref5)

pref5 <- as.character(pref5)

pref6 <- df %>% 
  filter(overall_rank == x) %>% 
  select(first_round_pref6)

pref6 <- as.character(pref6)

pref7 <- df %>% 
  filter(overall_rank == x) %>% 
  select(first_round_pref7)

pref7 <- as.character(pref7)

pref8 <- df %>% 
  filter(overall_rank == x) %>% 
  select(first_round_pref8)

pref8 <- as.character(pref8)

df <- df %>% 
  mutate(first_round_loc = case_when(
    overall_rank == x & pref1 %in% options ~ first_round_pref1,
    overall_rank == x & pref2 %in% options ~ first_round_pref2,
    overall_rank == x & pref3 %in% options ~ first_round_pref3,
    overall_rank == x & pref4 %in% options ~ first_round_pref4,
    overall_rank == x & pref5 %in% options ~ first_round_pref5,
    overall_rank == x & pref6 %in% options ~ first_round_pref6,
    overall_rank == x & pref7 %in% options ~ first_round_pref7,
    overall_rank == x & pref8 %in% options ~ first_round_pref8,
    TRUE ~ first_round_loc
  ))

options <- as.data.frame(options)
options <- data.frame(t(options)) %>% 
  mutate(loc = 0)

colnames(options) <- c("location", "loc")

df2 <- df %>% 
  filter(overall_rank == x)

options <- options %>% 
  mutate(loc = case_when(
    df2$first_round_loc == location ~ 1,
    TRUE ~ 0
  )) %>% 
  mutate(loc2 = ifelse(cumsum(loc) <= 1 & loc == 1, 1, 0)) %>% 
  filter(loc2 == 0) %>% 
  select(location)

# Team 18
pref1 <- df %>% 
  filter(overall_rank == x + 1) %>% 
  select(first_round_pref1)

pref1 <- as.character(pref1)

pref2 <- df %>% 
  filter(overall_rank == x + 1) %>% 
  select(first_round_pref2)

pref2 <- as.character(pref2)

pref3 <- df %>% 
  filter(overall_rank == x + 1) %>% 
  select(first_round_pref3)

pref3 <- as.character(pref3)

pref4 <- df %>% 
  filter(overall_rank == x + 1) %>% 
  select(first_round_pref4)

pref4 <- as.character(pref4)

pref5 <- df %>% 
  filter(overall_rank == x + 1) %>% 
  select(first_round_pref5)

pref5 <- as.character(pref5)

pref6 <- df %>% 
  filter(overall_rank == x + 1) %>% 
  select(first_round_pref6)

pref6 <- as.character(pref6)

pref7 <- df %>% 
  filter(overall_rank == x + 1) %>% 
  select(first_round_pref7)

pref7 <- as.character(pref7)

pref8 <- df %>% 
  filter(overall_rank == x + 1) %>% 
  select(first_round_pref8)

pref8 <- as.character(pref8)

options <- as.list(options$location)

df <- df %>% 
  mutate(first_round_loc = case_when(
    overall_rank == x + 1 & pref1 %in% options ~ first_round_pref1,
    overall_rank == x + 1 & pref2 %in% options ~ first_round_pref2,
    overall_rank == x + 1 & pref3 %in% options ~ first_round_pref3,
    overall_rank == x + 1 & pref4 %in% options ~ first_round_pref4,
    overall_rank == x + 1 & pref5 %in% options ~ first_round_pref5,
    overall_rank == x + 1 & pref6 %in% options ~ first_round_pref6,
    overall_rank == x + 1 & pref7 %in% options ~ first_round_pref7,
    overall_rank == x + 1 & pref8 %in% options ~ first_round_pref8,
    TRUE ~ first_round_loc
  ))

options <- as.data.frame(options)
options <- data.frame(t(options)) %>% 
  mutate(loc = 0)

colnames(options) <- c("location", "loc")

df2 <- df %>% 
  filter(overall_rank == x + 1)

options <- options %>% 
  mutate(loc = case_when(
    df2$first_round_loc == location ~ 1,
    TRUE ~ 0
  )) %>% 
  mutate(loc2 = ifelse(cumsum(loc) <= 1 & loc == 1, 1, 0)) %>% 
  filter(loc2 == 0) %>% 
  select(location)

# Team 19
pref1 <- df %>% 
  filter(overall_rank == x + 2) %>% 
  select(first_round_pref1)

pref1 <- as.character(pref1)

pref2 <- df %>% 
  filter(overall_rank == x + 2) %>% 
  select(first_round_pref2)

pref2 <- as.character(pref2)

pref3 <- df %>% 
  filter(overall_rank == x + 2) %>% 
  select(first_round_pref3)

pref3 <- as.character(pref3)

pref4 <- df %>% 
  filter(overall_rank == x + 2) %>% 
  select(first_round_pref4)

pref4 <- as.character(pref4)

pref5 <- df %>% 
  filter(overall_rank == x + 2) %>% 
  select(first_round_pref5)

pref5 <- as.character(pref5)

pref6 <- df %>% 
  filter(overall_rank == x + 2) %>% 
  select(first_round_pref6)

pref6 <- as.character(pref6)

pref7 <- df %>% 
  filter(overall_rank == x + 2) %>% 
  select(first_round_pref7)

pref7 <- as.character(pref7)

pref8 <- df %>% 
  filter(overall_rank == x + 2) %>% 
  select(first_round_pref8)

pref8 <- as.character(pref8)

options <- as.list(options$location)

df <- df %>% 
  mutate(first_round_loc = case_when(
    overall_rank == x + 2 & pref1 %in% options ~ first_round_pref1,
    overall_rank == x + 2 & pref2 %in% options ~ first_round_pref2,
    overall_rank == x + 2 & pref3 %in% options ~ first_round_pref3,
    overall_rank == x + 2 & pref4 %in% options ~ first_round_pref4,
    overall_rank == x + 2 & pref5 %in% options ~ first_round_pref5,
    overall_rank == x + 2 & pref6 %in% options ~ first_round_pref6,
    overall_rank == x + 2 & pref7 %in% options ~ first_round_pref7,
    overall_rank == x + 2 & pref8 %in% options ~ first_round_pref8,
    TRUE ~ first_round_loc
  ))

options <- as.data.frame(options)
options <- data.frame(t(options)) %>% 
  mutate(loc = 0)

colnames(options) <- c("location", "loc")

df2 <- df %>% 
  filter(overall_rank == x + 2)

options <- options %>% 
  mutate(loc = case_when(
    df2$first_round_loc == location ~ 1,
    TRUE ~ 0
  )) %>% 
  mutate(loc2 = ifelse(cumsum(loc) <= 1 & loc == 1, 1, 0)) %>% 
  filter(loc2 == 0) %>% 
  select(location)

# Team 20
pref1 <- df %>% 
  filter(overall_rank == x + 3) %>% 
  select(first_round_pref1)

pref1 <- as.character(pref1)

pref2 <- df %>% 
  filter(overall_rank == x + 3) %>% 
  select(first_round_pref2)

pref2 <- as.character(pref2)

pref3 <- df %>% 
  filter(overall_rank == x + 3) %>% 
  select(first_round_pref3)

pref3 <- as.character(pref3)

pref4 <- df %>% 
  filter(overall_rank == x + 3) %>% 
  select(first_round_pref4)

pref4 <- as.character(pref4)

pref5 <- df %>% 
  filter(overall_rank == x + 3) %>% 
  select(first_round_pref5)

pref5 <- as.character(pref5)

pref6 <- df %>% 
  filter(overall_rank == x + 3) %>% 
  select(first_round_pref6)

pref6 <- as.character(pref6)

pref7 <- df %>% 
  filter(overall_rank == x + 3) %>% 
  select(first_round_pref7)

pref7 <- as.character(pref7)

pref8 <- df %>% 
  filter(overall_rank == x + 3) %>% 
  select(first_round_pref8)

pref8 <- as.character(pref8)

options <- as.list(options$location)

df <- df %>% 
  mutate(first_round_loc = case_when(
    overall_rank == x + 3 & pref1 %in% options ~ first_round_pref1,
    overall_rank == x + 3 & pref2 %in% options ~ first_round_pref2,
    overall_rank == x + 3 & pref3 %in% options ~ first_round_pref3,
    overall_rank == x + 3 & pref4 %in% options ~ first_round_pref4,
    overall_rank == x + 3 & pref5 %in% options ~ first_round_pref5,
    overall_rank == x + 3 & pref6 %in% options ~ first_round_pref6,
    overall_rank == x + 3 & pref7 %in% options ~ first_round_pref7,
    overall_rank == x + 3 & pref8 %in% options ~ first_round_pref8,
    TRUE ~ first_round_loc
  ))

options <- as.data.frame(options)
options <- data.frame(t(options)) %>% 
  mutate(loc = 0)

colnames(options) <- c("location", "loc")

df2 <- df %>% 
  filter(overall_rank == x + 3)

options <- options %>% 
  mutate(loc = case_when(
    df2$first_round_loc == location ~ 1,
    TRUE ~ 0
  )) %>% 
  mutate(loc2 = ifelse(cumsum(loc) <= 1 & loc == 1, 1, 0)) %>% 
  filter(loc2 == 0) %>% 
  select(location)

  return (df)
}


# 5, 12, and 13 seeds
option1 <- all_tourney_teams_loc %>% 
  filter(overall_rank == 13) %>% 
  select(first_round_loc)

option2 <- all_tourney_teams_loc %>% 
  filter(overall_rank == 14) %>% 
  select(first_round_loc)

option3 <-  all_tourney_teams_loc %>% 
  filter(overall_rank == 15) %>% 
  select(first_round_loc)

option4 <- all_tourney_teams_loc %>% 
  filter(overall_rank == 16) %>% 
  select(first_round_loc)

options <- c(option1, option2, option3, option4)


all_tourney_teams_loc <- first_round_cities(all_tourney_teams_loc, 17)
all_tourney_teams_loc <- first_round_cities(all_tourney_teams_loc, 47)
all_tourney_teams_loc <- first_round_cities(all_tourney_teams_loc, 51)

# 6, 11, and 14 seeds
option1 <- all_tourney_teams_loc %>% 
  filter(overall_rank == 9) %>% 
  select(first_round_loc)

option2 <- all_tourney_teams_loc %>% 
  filter(overall_rank == 10) %>% 
  select(first_round_loc)

option3 <-  all_tourney_teams_loc %>% 
  filter(overall_rank == 11) %>% 
  select(first_round_loc)

option4 <- all_tourney_teams_loc %>% 
  filter(overall_rank == 12) %>% 
  select(first_round_loc)

options <- c(option1, option2, option3, option4)


all_tourney_teams_loc <- first_round_cities(all_tourney_teams_loc, 21)
all_tourney_teams_loc <- first_round_cities(all_tourney_teams_loc, 43) # needs to be 41 when First Four is 11 seeds, its 43 for 10 seeds
all_tourney_teams_loc <- first_round_cities(all_tourney_teams_loc, 55)

# 7, 10, and 15 seeds
option1 <- all_tourney_teams_loc %>% 
  filter(overall_rank == 5) %>% 
  select(first_round_loc)

option2 <- all_tourney_teams_loc %>% 
  filter(overall_rank == 6) %>% 
  select(first_round_loc)

option3 <-  all_tourney_teams_loc %>% 
  filter(overall_rank == 7) %>% 
  select(first_round_loc)

option4 <- all_tourney_teams_loc %>% 
  filter(overall_rank == 8) %>% 
  select(first_round_loc)

options <- c(option1, option2, option3, option4)


all_tourney_teams_loc <- first_round_cities(all_tourney_teams_loc, 25)
all_tourney_teams_loc <- first_round_cities(all_tourney_teams_loc, 37)
all_tourney_teams_loc <- first_round_cities(all_tourney_teams_loc, 59)

# 8, 9, and 16 seeds
option1 <- all_tourney_teams_loc %>% 
  filter(overall_rank == 1) %>% 
  select(first_round_loc)

option2 <- all_tourney_teams_loc %>% 
  filter(overall_rank == 2) %>% 
  select(first_round_loc)

option3 <-  all_tourney_teams_loc %>% 
  filter(overall_rank == 3) %>% 
  select(first_round_loc)

option4 <- all_tourney_teams_loc %>% 
  filter(overall_rank == 4) %>% 
  select(first_round_loc)

options <- c(option1, option2, option3, option4)


all_tourney_teams_loc <- first_round_cities(all_tourney_teams_loc, 29)
all_tourney_teams_loc <- first_round_cities(all_tourney_teams_loc, 33)
all_tourney_teams_loc <- first_round_cities(all_tourney_teams_loc, 63)

# First Four Teams
# 42 and 43 play each other so same region
# 44 and 45 play each other
# Have to replace 43 with 42nd's location
# 44 and 45 gets 43 original's location
# Right now 46 is an auto bid non-power champion
# Give 46 44's original region

# UPDATED WITH 46 AT LARGE TEAMS
# 43 and 44 play each, 45 and 46 play each other
# 45 and 46 needs to get 44's original location
# 44 needs to get 43's location

# UPDATED WITH 44 AT LARGE TEAMS
# 41 and 42 play each other
# 43 and 44 play each other
# 42 needs 41's location
# 44 needs 43's location
# 46 needs 44's original location
# 45 needs 43's original location

# UPDATED WITH 42 AT LARGE TEAMS
# 39 and 40 play each other
# 41 and 42 play each other
# 40 needs 39's location
# 42 and 41 need 40's location

# all_tourney_teams_loc <- all_tourney_teams_loc %>% 
#   mutate(first_round_loc = case_when(
#     overall_rank == 46 ~ lag(first_round_loc, n = 2),
#     TRUE ~ first_round_loc
#   ))
# 
# all_tourney_teams_loc <- all_tourney_teams_loc %>% 
#   mutate(first_round_loc = case_when(
#     overall_rank == 45 ~ lag(first_round_loc, n = 2),
#     TRUE ~ first_round_loc
#   ))
# 
# all_tourney_teams_loc <- all_tourney_teams_loc %>% 
#   mutate(first_round_loc = case_when(
#     overall_rank == 44 ~ lag(first_round_loc, n = 1),
#     TRUE ~ first_round_loc
#   ))
# 
# 
# 
# all_tourney_teams_loc <- all_tourney_teams_loc %>% 
#   mutate(first_round_loc = case_when(
#     overall_rank == 43 ~ lag(first_round_loc),
#    TRUE ~ first_round_loc
#   ))

all_tourney_teams_loc <- all_tourney_teams_loc %>% 
  mutate(first_round_loc = case_when(
    overall_rank == 42 ~ lag(first_round_loc, n = 2),
    TRUE ~ first_round_loc
  ))

all_tourney_teams_loc <- all_tourney_teams_loc %>% 
  mutate(first_round_loc = case_when(
    overall_rank == 41 ~ lag(first_round_loc),
    TRUE ~ first_round_loc
  ))

all_tourney_teams_loc <- all_tourney_teams_loc %>% 
  mutate(first_round_loc = case_when(
    overall_rank == 40 ~ lag(first_round_loc),
    TRUE ~ first_round_loc
  ))

# 16 seeds 
all_tourney_teams_loc <- all_tourney_teams_loc %>% 
  mutate(first_round_loc = case_when(
    overall_rank == 67 ~ lag(first_round_loc, n = 1),
    TRUE ~ first_round_loc
  ))

all_tourney_teams_loc <- all_tourney_teams_loc %>% 
  mutate(first_round_loc = case_when(
    overall_rank == 68 ~ lag(first_round_loc, n = 1),
    TRUE ~ first_round_loc
  ))

all_tourney_teams_loc <- all_tourney_teams_loc %>% 
  mutate(first_round_loc = case_when(
    overall_rank == 66 ~ lag(first_round_loc),
    TRUE ~ first_round_loc
  ))

all_tourney_teams_loc2 <- all_tourney_teams_loc %>% 
  select(Team, overall_rank, seed, first_four,
         region, first_round_loc)

# Assign 5-16 seeds a region based on first round location of seeds 1-4

rest_of_locations <- function(x, n1, order) {

 if (order == 1) {
all_tourney_teams_loc2 <- all_tourney_teams_loc2 %>% 
  mutate(region = case_when(
    overall_rank == x & first_round_loc == lag(first_round_loc, n = n1) ~ lag(region, n = n1),
    overall_rank == x & first_round_loc == lag(first_round_loc, n = n1 - 1) ~ lag(region, n = n1 - 1),
    overall_rank == x & first_round_loc == lag(first_round_loc, n = n1 - 2) ~ lag(region, n = n1 - 2),
    overall_rank == x & first_round_loc == lag(first_round_loc, n = n1 - 3) ~ lag(region, n = n1 - 3),
    TRUE ~ region
  ))
 } else if (order == 2) {
   all_tourney_teams_loc2 <- all_tourney_teams_loc2 %>% 
     mutate(region = case_when(
       overall_rank == x & first_round_loc == lag(first_round_loc, n = n1) & lag(region, n = 1) != lag(region, n = n1)
       ~ lag(region, n = n1),
       overall_rank == x & first_round_loc == lag(first_round_loc, n = n1 - 1) & lag(region, n = 1) != lag(region, n = n1 - 1)
       ~ lag(region, n = n1 - 1),
       overall_rank == x & first_round_loc == lag(first_round_loc, n = n1 - 2) & lag(region, n = 1) != lag(region, n = n1 - 2)
       ~ lag(region, n = n1 - 2),
       overall_rank == x & first_round_loc == lag(first_round_loc, n = n1 - 3) & lag(region, n = 1) != lag(region, n = n1 - 3)
       ~ lag(region, n = n1 - 3),
       TRUE ~ region
     ))
 } else if (order == 3) {
   all_tourney_teams_loc2 <- all_tourney_teams_loc2 %>% 
     mutate(region = case_when(
       overall_rank == x & first_round_loc == lag(first_round_loc, n = n1) & lag(region, n = 1) != lag(region, n = n1)
       & lag(region, n = 2) != lag(region, n = n1) ~ lag(region, n = n1),
       overall_rank == x & first_round_loc == lag(first_round_loc, n = n1 - 1) & lag(region, n = 1) != lag(region, n = n1 - 1)
       & lag(region, n = 2) != lag(region, n = n1 -1) ~ lag(region, n = n1 - 1),
       overall_rank == x & first_round_loc == lag(first_round_loc, n = n1 - 2) & lag(region, n = 1) != lag(region, n = n1 - 2)
       & lag(region, n = 2) != lag(region, n = n1 -2) ~ lag(region, n = n1 - 2),
       overall_rank == x & first_round_loc == lag(first_round_loc, n = n1 - 3) & lag(region, n = 1) != lag(region, n = n1 - 3)
       & lag(region, n = 2) != lag(region, n = n1 - 3) ~ lag(region, n = n1 - 3),
       TRUE ~ region
     ))
 } else if (order == 4) {
   all_tourney_teams_loc2 <- all_tourney_teams_loc2 %>% 
     mutate(region = case_when(
       overall_rank == x & first_round_loc == lag(first_round_loc, n = n1) & lag(region, n = 1) != lag(region, n = n1)
       & lag(region, n = 2) != lag(region, n = n1) & lag(region, n = 3) != lag(region, n = n1) ~ lag(region, n = n1),
       overall_rank == x & first_round_loc == lag(first_round_loc, n = n1 - 1) & lag(region, n = 1) != lag(region, n = n1 - 1)
       & lag(region, n = 2) != lag(region, n = n1 -1) & lag(region, n = 3) != lag(region, n = n1 -1) ~ lag(region, n = n1 - 1),
       overall_rank == x & first_round_loc == lag(first_round_loc, n = n1 - 2) & lag(region, n = 1) != lag(region, n = n1 - 2)
       & lag(region, n = 2) != lag(region, n = n1 -2) & lag(region, n = 3) != lag(region, n = n1 -2)~ lag(region, n = n1 - 2),
       overall_rank == x & first_round_loc == lag(first_round_loc, n = n1 - 3) & lag(region, n = 1) != lag(region, n = n1 - 3)
       & lag(region, n = 2) != lag(region, n = n1 - 3) & lag(region, n = 3) != lag(region, n = n1 -3)~ lag(region, n = n1 - 3),
       TRUE ~ region
     ))
  }

}

# 5 Seeds
all_tourney_teams_loc2 <- rest_of_locations(17, 4, 1)
all_tourney_teams_loc2 <- rest_of_locations(18, 5, 2)
all_tourney_teams_loc2 <- rest_of_locations(19, 6, 3)
all_tourney_teams_loc2 <- rest_of_locations(20, 7, 4)
# 6 Seeds
all_tourney_teams_loc2 <- rest_of_locations(21, 12, 1)
all_tourney_teams_loc2 <- rest_of_locations(22, 13, 2)
all_tourney_teams_loc2 <- rest_of_locations(23, 14, 3)
all_tourney_teams_loc2 <- rest_of_locations(24, 15, 4)
# 7 Seeds
all_tourney_teams_loc2 <- rest_of_locations(25, 20, 1)
all_tourney_teams_loc2 <- rest_of_locations(26, 21, 2)
all_tourney_teams_loc2 <- rest_of_locations(27, 22, 3)
all_tourney_teams_loc2 <- rest_of_locations(28, 23, 4)
# 8 Seeds
all_tourney_teams_loc2 <- rest_of_locations(29, 28, 1)
all_tourney_teams_loc2 <- rest_of_locations(30, 29, 2)
all_tourney_teams_loc2 <- rest_of_locations(31, 30, 3)
all_tourney_teams_loc2 <- rest_of_locations(32, 31, 4)
# 9 Seeds
all_tourney_teams_loc2 <- rest_of_locations(33, 32, 1)
all_tourney_teams_loc2 <- rest_of_locations(34, 33, 2)
all_tourney_teams_loc2 <- rest_of_locations(35, 34, 3)
all_tourney_teams_loc2 <- rest_of_locations(36, 35, 4)
# 10 Seeds
all_tourney_teams_loc2 <- rest_of_locations(37, 32, 1)
all_tourney_teams_loc2 <- rest_of_locations(38, 33, 2)
all_tourney_teams_loc2 <- rest_of_locations(39, 34, 3)
all_tourney_teams_loc2 <- rest_of_locations(40, 35, 4)
# 11 Seeds - not sure if this is right
all_tourney_teams_loc2 <- rest_of_locations(41, 32, 1)
all_tourney_teams_loc2 <- rest_of_locations(42, 33, 2)
all_tourney_teams_loc2 <- rest_of_locations(43, 34, 2)
all_tourney_teams_loc2 <- rest_of_locations(44, 35, 3)
all_tourney_teams_loc2 <- rest_of_locations(45, 36, 3)
all_tourney_teams_loc2 <- rest_of_locations(46, 37, 4)
# 12 Seeds
all_tourney_teams_loc2 <- rest_of_locations(47, 34, 1)
all_tourney_teams_loc2 <- rest_of_locations(48, 35, 2)
all_tourney_teams_loc2 <- rest_of_locations(49, 36, 3)
all_tourney_teams_loc2 <- rest_of_locations(50, 37, 4)
# 13 Seeds
all_tourney_teams_loc2 <- rest_of_locations(51, 38, 1)
all_tourney_teams_loc2 <- rest_of_locations(52, 39, 2)
all_tourney_teams_loc2 <- rest_of_locations(53, 40, 3)
all_tourney_teams_loc2 <- rest_of_locations(54, 41, 4)
# 14 Seeds
all_tourney_teams_loc2 <- rest_of_locations(55, 46, 1)
all_tourney_teams_loc2 <- rest_of_locations(56, 47, 2)
all_tourney_teams_loc2 <- rest_of_locations(57, 48, 3)
all_tourney_teams_loc2 <- rest_of_locations(58, 49, 4)
# 15 Seeds
all_tourney_teams_loc2 <- rest_of_locations(59, 54, 1)
all_tourney_teams_loc2 <- rest_of_locations(60, 55, 2)
all_tourney_teams_loc2 <- rest_of_locations(61, 56, 3)
all_tourney_teams_loc2 <- rest_of_locations(62, 57, 4)
# 16 Seeds
all_tourney_teams_loc2 <- rest_of_locations(63, 62, 1)
all_tourney_teams_loc2 <- rest_of_locations(64, 63, 2)
all_tourney_teams_loc2 <- rest_of_locations(65, 64, 3)
all_tourney_teams_loc2 <- rest_of_locations(66, 65, 3)
all_tourney_teams_loc2 <- rest_of_locations(67, 66, 4)
all_tourney_teams_loc2 <- rest_of_locations(68, 67, 4)

# Fix the first four teams
all_tourney_teams_loc2 <- all_tourney_teams_loc2 %>% 
  mutate(region = case_when(
    overall_rank == 40 ~ lag(region),
    overall_rank == 42 ~ lag(region),
    overall_rank == 66 ~ lag(region),
    overall_rank == 68 ~ lag(region),
    TRUE ~ region
  ))
}

#### Fix conferences playing each other in first round ##########
# switch A&M and Indiana St.
all_tourney_teams_loc2[40, 1] <- "Texas A&M"
all_tourney_teams_loc2[42, 1] <- "Indiana St."

# new mexico and northwestern
all_tourney_teams_loc2[33, 5] <- "LA"
all_tourney_teams_loc2[33, 6] <- "charlotte"
all_tourney_teams_loc2[35, 5] <- "Dallas"
all_tourney_teams_loc2[35, 6] <- "memphis"


# byu and wisconsin 1st round location fix
all_tourney_teams_loc2[17, 6] <- "spokane"
all_tourney_teams_loc2[19, 6] <- "pitt"
all_tourney_teams_loc2[13, 6] <- "spokane"
all_tourney_teams_loc2[15, 6] <- "pitt"
all_tourney_teams_loc2[47, 6] <- "spokane"
all_tourney_teams_loc2[49, 6] <- "pitt"
all_tourney_teams_loc2[51, 6] <- "spokane"
all_tourney_teams_loc2[53, 6] <- "pitt"

# texas and nevada
all_tourney_teams_loc2[25, 5] <- "Boston"
all_tourney_teams_loc2[25, 6] <- "indy"
all_tourney_teams_loc2[27, 5] <- "LA"
all_tourney_teams_loc2[27, 6] <- "saltlake"

# sdsu and nevada
all_tourney_teams_loc2[28, 5] <- "Boston"
all_tourney_teams_loc2[28, 6] <- "indy"
all_tourney_teams_loc2[25, 5] <- "Dallas"
all_tourney_teams_loc2[25, 6] <- "charlotte"


# Manually fix the blank ones cause i can't figure it out
#  will have to change this every time
all_tourney_teams_loc2[41,5] <- "Dallas"
all_tourney_teams_loc2[42,5] <- "Dallas"

# Make the bracket
{
# Get the regions
east <- all_tourney_teams_loc2 %>% 
  filter(region == "Boston")

south <- all_tourney_teams_loc2 %>% 
  filter(region == "Dallas")

midwest <- all_tourney_teams_loc2 %>% 
  filter(region == "Detroit")

west <- all_tourney_teams_loc2 %>% 
  filter(region == "LA")


east_table <- east %>% 
  filter(seed == 1)

east_table <- east_table %>% 
  rbind(east %>% filter(seed == 16)) %>% 
  rbind(east %>% filter(seed == 8)) %>% 
  rbind(east %>% filter(seed == 9)) %>% 
  rbind(east %>% filter(seed == 4)) %>% 
  rbind(east %>% filter(seed == 13)) %>% 
  rbind(east %>% filter(seed == 5)) %>% 
  rbind(east %>% filter(seed == 12)) %>% 
  rbind(east %>% filter(seed == 6)) %>% 
  rbind(east %>% filter(seed == 11)) %>% 
  rbind(east %>% filter(seed == 3)) %>% 
  rbind(east %>% filter(seed == 14)) %>% 
  rbind(east %>% filter(seed == 7)) %>% 
  rbind(east %>% filter(seed == 10)) %>% 
  rbind(east %>% filter(seed == 2)) %>% 
  rbind(east %>% filter(seed == 15))



east_table <- east_table %>% 
  mutate(group = case_when(
    seed %in% c(1,8,9,16) ~ " ",
    seed %in% c(4,5,12,13) ~ "",
    seed %in% c(6,11,3,14) ~ " ",
    seed %in% c(7,10,2,15) ~ "",
    TRUE ~ "NA"
  ))

# get logos and then group_by
# change back to 43 and 45
east_bracket <- east_table %>%
  cbbplotR::gt_cbb_teams(Team, Team) %>% 
  group_by(group, first_round_loc) %>% 
  mutate(Team = ifelse(first_four == 1 & seed == lag(seed), str_c(Team, " / ", lag(Team)), Team)) %>% 
  filter(!(overall_rank %in% c(39,41,65,67))) %>% 
  select(seed, Team) %>% 
  gt() %>% 
  gt::fmt_markdown(Team) %>%
  gt::cols_align(columns = Team, 'left') %>% 
  tab_options(column_labels.hidden = TRUE) %>% 
  tab_header(title = "EAST Region") %>% 
  as_raw_html()




# Midwest
midwest_table <- midwest %>% 
  filter(seed == 1)

midwest_table <- midwest_table %>% 
  rbind(midwest %>% filter(seed == 16)) %>% 
  rbind(midwest %>% filter(seed == 8)) %>% 
  rbind(midwest %>% filter(seed == 9)) %>% 
  rbind(midwest %>% filter(seed == 4)) %>% 
  rbind(midwest %>% filter(seed == 13)) %>% 
  rbind(midwest %>% filter(seed == 5)) %>% 
  rbind(midwest %>% filter(seed == 12)) %>% 
  rbind(midwest %>% filter(seed == 6)) %>% 
  rbind(midwest %>% filter(seed == 11)) %>% 
  rbind(midwest %>% filter(seed == 3)) %>% 
  rbind(midwest %>% filter(seed == 14)) %>% 
  rbind(midwest %>% filter(seed == 7)) %>% 
  rbind(midwest %>% filter(seed == 10)) %>% 
  rbind(midwest %>% filter(seed == 2)) %>% 
  rbind(midwest %>% filter(seed == 15))



midwest_table <- midwest_table %>% 
  mutate(group = case_when(
    seed %in% c(1,8,9,16) ~ " ",
    seed %in% c(4,5,12,13) ~ "",
    seed %in% c(6,11,3,14) ~ " ",
    seed %in% c(7,10,2,15) ~ "",
    TRUE ~ "NA"
  ))


midwest_bracket <- midwest_table %>%
  cbbplotR::gt_cbb_teams(Team, Team) %>% 
  group_by(group, first_round_loc) %>% 
  mutate(Team = ifelse(first_four == 1 & seed == lag(seed), str_c(Team, " / ", lag(Team)), Team)) %>% 
  filter(!(overall_rank %in% c(39,41,65,67))) %>% 
  select(seed, Team) %>% 
  gt() %>% 
  gt::fmt_markdown(Team) %>%
  gt::cols_align(columns = Team, 'left') %>% 
  tab_options(column_labels.hidden = TRUE) %>% 
  tab_header(title = "Midwest Region") %>% 
  as_raw_html()

# South
south_table <- south %>% 
  filter(seed == 1)

south_table <- south_table %>% 
  rbind(south %>% filter(seed == 16)) %>% 
  rbind(south %>% filter(seed == 8)) %>% 
  rbind(south %>% filter(seed == 9)) %>% 
  rbind(south %>% filter(seed == 4)) %>% 
  rbind(south %>% filter(seed == 13)) %>% 
  rbind(south %>% filter(seed == 5)) %>% 
  rbind(south %>% filter(seed == 12)) %>% 
  rbind(south %>% filter(seed == 6)) %>% 
  rbind(south %>% filter(seed == 11)) %>% 
  rbind(south %>% filter(seed == 3)) %>% 
  rbind(south %>% filter(seed == 14)) %>% 
  rbind(south %>% filter(seed == 7)) %>% 
  rbind(south %>% filter(seed == 10)) %>% 
  rbind(south %>% filter(seed == 2)) %>% 
  rbind(south %>% filter(seed == 15))



south_table <- south_table %>% 
  mutate(group = case_when(
    seed %in% c(1,8,9,16) ~ "",
    seed %in% c(4,5,12,13) ~ "",
    seed %in% c(6,11,3,14) ~ " ",
    seed %in% c(7,10,2,15) ~ " ",
    TRUE ~ "NA"
  ))



south_bracket <- south_table %>%
  cbbplotR::gt_cbb_teams(Team, Team) %>% 
  group_by(group, first_round_loc) %>% 
  mutate(Team = ifelse(first_four == 1 & seed == lag(seed), str_c(Team, " / ", lag(Team)), Team)) %>% 
  filter(!(overall_rank %in% c(39,41,65,67))) %>% 
  select(seed, Team) %>% 
  gt() %>% 
  gt::fmt_markdown(Team) %>%
  gt::cols_align(columns = Team, 'left') %>% 
  tab_options(column_labels.hidden = TRUE) %>% 
  tab_header(title = "SOUTH Region") %>% 
  as_raw_html()


# West
west_table <- west %>% 
  filter(seed == 1)

west_table <- west_table %>% 
  rbind(west %>% filter(seed == 16)) %>% 
  rbind(west %>% filter(seed == 8)) %>% 
  rbind(west %>% filter(seed == 9)) %>% 
  rbind(west %>% filter(seed == 4)) %>% 
  rbind(west %>% filter(seed == 13)) %>% 
  rbind(west %>% filter(seed == 5)) %>% 
  rbind(west %>% filter(seed == 12)) %>% 
  rbind(west %>% filter(seed == 6)) %>% 
  rbind(west %>% filter(seed == 11)) %>% 
  rbind(west %>% filter(seed == 3)) %>% 
  rbind(west %>% filter(seed == 14)) %>% 
  rbind(west %>% filter(seed == 7)) %>% 
  rbind(west %>% filter(seed == 10)) %>% 
  rbind(west %>% filter(seed == 2)) %>% 
  rbind(west %>% filter(seed == 15))



west_table <- west_table %>% 
  mutate(group = case_when(
    seed %in% c(1,8,9,16) ~ " ",
    seed %in% c(4,5,12,13) ~ " ",
    seed %in% c(6,11,3,14) ~ "",
    seed %in% c(7,10,2,15) ~ " ",
    TRUE ~ "NA"
  ))


west_bracket <- west_table %>%
  cbbplotR::gt_cbb_teams(Team, Team) %>% 
  group_by(group, first_round_loc) %>% 
  mutate(Team = ifelse(first_four == 1 & seed == lag(seed), str_c(Team, " / ", lag(Team)), Team)) %>% 
  filter(!(overall_rank %in% c(39,41,65,67))) %>% 
  select(seed, Team) %>% 
  gt() %>% 
  gt::fmt_markdown(Team) %>%
  gt::cols_align(columns = Team, 'left') %>% 
  tab_options(column_labels.hidden = TRUE) %>% 
  tab_header(title = "WEST Region") %>% 
  as_raw_html()

bracket <- data.frame(south = south_bracket,
                      east = east_bracket,
                      midwest = midwest_bracket,
                      west = west_bracket)


final_bracket <- bracket %>%
  gt() %>% 
  fmt_markdown(columns = TRUE) %>% 
  cols_label(midwest = "Midwest", south = "South",
             west = "West", east = "East") %>% 
  tab_options(column_labels.hidden = TRUE) %>% 
  tab_header(title = "AA Bracketology",
             subtitle = "Final Bracket") %>% 
  opt_stylize(style = 1, color = "blue")

final_bracket
}

gtsave(final_bracket, "final_bracket.png", expand = 100, vwidth = 1800, vheight = 3200)  


write_csv(all_tourney_teams_loc2, "aabracketology_final_seeds.csv")
