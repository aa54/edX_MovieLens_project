################################
# Create edx set, validation set
################################
install.packages("tidyverse")
install.packages("caret")
install.packages("data.table")
library("tidyverse")
library("data.table")
library("caret")
library("dplyr")

# libraries needed for data analysis and visualization
library(stringr)
library(lubridate)
library(ggplot2)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)
ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))
movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% 
  mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))
movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)
rm(dl, ratings, movies, test_index, temp, movielens, removed)

#######################
#   Data Pre-processing
#######################

# take care of any missing values in the datasets
edx %>% sapply(., function(x) sum(is.na(x)))
validation %>% sapply(., function(x) sum(is.na(x)))

# extract year from title and clean title
edx <- edx %>%
  mutate(title = str_trim(title)) %>%
  # split title to title, year
  extract(title, c("title_tmp", "year"), regex = "^(.*) \\(([0-9 \\-]*)\\)$", remove = F) %>%
  mutate(year = if_else(str_length(year) > 4, as.integer(str_split(year, "-", simplify = T)[1]), as.integer(year))) %>%
  # replace title NA's with original title
  mutate(title = if_else(is.na(title_tmp), title, title_tmp)) %>%
  # drop title_tmp column
  select(-title_tmp)
validation <- validation %>%
  mutate(title = str_trim(title)) %>%
  extract(title, c("title_tmp", "year"), regex = "^(.*) \\(([0-9 \\-]*)\\)$", remove = F) %>%
  mutate(year = if_else(str_length(year) > 4, as.integer(str_split(year, "-", simplify = T)[1]), as.integer(year))) %>%
  mutate(title = if_else(is.na(title_tmp), title, title_tmp)) %>%
  select(-title_tmp)

# looks like no other pre-processing cleanup is needed for now

#################################
# Data Analysis and Visualization
#################################

# general stats
head(edx)
head(validation)

# summary stats
summary(edx)
summary(validation)

# number of unique users and movies
edx %>% 
  summarize(n_users = n_distinct(userId),
            n_movies = n_distinct(movieId))

# ratings distribution - check unique values and plot the distirbutions
sort(unique(edx$rating))
ggplot(edx, aes(rating)) + 
  geom_bar(color="orange") +
  ggtitle("Ratings Distribution") +
  xlab("Rating") + ylab("Counts")

# distribution of users rating movies
edx %>% count(userId) %>%
  ggplot(aes(n)) +
  geom_histogram(bins=25, color="black") +
  scale_x_log10() +
  ggtitle("Users") +
  xlab("No. of users") + ylab("No. of movies rated")

edx %>% count(movieId) %>%
  ggplot(aes(n)) +
  geom_histogram(bins=25, color="black") +
  scale_x_log10() +
  ggtitle("Movies") +
  xlab("No. of users") + ylab("No. of movies rated")

###################
# Model preparation
###################

# RMSE function for vectors of ratings and corresponding pedictors
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

###################################
#  Baseline Model (Just the Average)
###################################

# get the average of all ratings and view
mu_hat <- mean(edx$rating)
mu_hat

# run the model
model_1_rmse <- RMSE(validation$rating, mu_hat)

# store the model outcome and view
rmse_results <- tibble(method = "Baseline Model", RMSE = model_1_rmse)
rmse_results %>% knitr::kable()

####################
# Movie Effect Model
####################

# get the average rating 
mu <- mean(edx$rating)
# get the least square estimates for each movie
movie_avgs <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

# view the variability of the estimates
qplot(b_i, data = movie_avgs, bins = 10, color = I("black"))

#generate the predictions set and run the model
predicted_ratings <- mu + validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  pull(b_i)
model_2_rmse <- RMSE(predicted_ratings, validation$rating)

# store the model outcome and view
rmse_results <- bind_rows(rmse_results, tibble(method = "Movie Effect Model", RMSE = model_2_rmse))
rmse_results %>% knitr::kable()

##############################
# Movie and User Effects Model
##############################

# compute the least square estimates for each user rating
user_avgs <- edx %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

# run the model
predicted_ratings <- validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

# model outcome
model_3_rmse <- RMSE(predicted_ratings, validation$rating)

# store the model outcome and view
rmse_results <- bind_rows(rmse_results, 
                          tibble(method = "Movie and User Effects Model", RMSE = model_3_rmse))
rmse_results %>% knitr::kable()

##########################################
# Regularized Movie and User Effects Model
##########################################

# build a vector with tuning parameters
lambdas <- seq(0, 10, 0.25)

# run the model
model_rmses <- sapply(lambdas, function(l){
  # first comute the regularized estimates for movies and users
  mu <- mean(edx$rating)
  b_i <- edx %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  b_u <- edx %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  # run the predictions
  predicted_ratings <- 
    validation %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  # run the model for each of the tuning parametrs
  return(RMSE(predicted_ratings, validation$rating))
})

# plot to view the results for all the tuning parametrs
qplot(lambdas, model_rmses)  

#chose the optimal tuning parameter for final compilation
lambda <- lambdas[which.min(model_rmses)]
lambda

# store the model outcome and view
rmse_results <- bind_rows(rmse_results, tibble(method = "Regularized Movie + User Effects Model", RMSE = min(model_rmses)))
rmse_results %>% knitr::kable()

