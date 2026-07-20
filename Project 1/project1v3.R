library(DBI)
library(RSQLite)

# This is the database connection for the raw data file
sales <- dbConnect(SQLite(), 
                   dbname = "project1_raw_data.db", 
                   flags = SQLITE_RO)

# List all tables to verify successful connection
dbListTables(sales)


##SECTION A##
#Step 1
SQL_train_head <- dbGetQuery(sales, "SELECT * FROM train 
                                LIMIT 5")
top_3_products <- dbGetQuery(sales, "SELECT item_nbr, SUM(units)
                             FROM train 
                             GROUP BY item_nbr
                             ORDER BY SUM(units) DESC
                             LIMIT 3")

#step 2- I replaced 'ON' with 'USING' in the join as I kept getting duplicate columns
#The code here will be repeated in step 3 as I cannot reference 'joined tables' within my SQL statements
#To ensure that my main table for analysis is relevant to the business question I will filter the table to only show the values relevant to the top 3 products( 5, 9 and 45)
joined_tables <- dbGetQuery(sales, "
                              SELECT * FROM train 
                              INNER JOIN key 
                                  USING (store_nbr) 
                              INNER JOIN weather 
                                  USING (station_nbr, date)
                              WHERE item_nbr IN (45, 9 ,5)
                              ")

View(joined_tables)

#task 3- for this task I will reference a top product using its ID(5)
weather_head <- dbGetQuery(sales, "SELECT * FROM weather 
                                LIMIT 5")

#Select statement only includes columns relevant to the task.
#I will deal with missing weather data in greater detail in later tasks but for the SQL task I will filter out M so I get no NA values in my output here.
sales_temp_5 <- dbGetQuery(sales, "SELECT date, AVG(CAST(tavg AS NUMERIC)) AS tavg, SUM(units) AS total_units 
                                  FROM train 
                                  INNER JOIN key 
                                      USING (store_nbr) 
                                  INNER JOIN weather 
                                      USING (station_nbr, date) 
                                  WHERE item_nbr = 5 AND tavg != 'M'
                                  GROUP BY date
                                  ORDER BY date ASC
                                  ")

View(sales_temp_5)

#============================================================
#SECTION B
#============================================================

library(readr)
library(dplyr)
library(tidyr)
library(lubridate)
library(ggplot2)

#checking the initial structure of table
summary(joined_tables)

# --- FIXING THE SPECIAL CHARACTERS & CODESUM CONTRADICTION ---
# The brief wants to know how 'T' (trace) values are handled. 
# I am removing whitespace and changing 'T' to a small number (0.005) so it doesn't break the medians later in my code.
joined_tables$preciptotal <- trimws(joined_tables$preciptotal)
joined_tables$preciptotal[joined_tables$preciptotal == "T"] <- "0.005"
joined_tables$preciptotal <- as.numeric(joined_tables$preciptotal)

#doing the same for the snowfall column
joined_tables$snowfall <- trimws(joined_tables$snowfall)
joined_tables$snowfall[joined_tables$snowfall == "T"] <- "0.005"
joined_tables$snowfall <- as.numeric(joined_tables$snowfall)

#tavg also needs to be numeric so it can be used as our primary model predictor
joined_tables$tavg <- as.numeric(joined_tables$tavg)

# EXPLICITLY Verifying data types and missing values to satisfy brief requirements
print(paste("Precipitation Class:", class(joined_tables$preciptotal)))
print(paste("Precipitation Missing Count:", sum(is.na(joined_tables$preciptotal))))
print(paste("Snowfall Class:", class(joined_tables$snowfall)))
print(paste("Temperature Class:", class(joined_tables$tavg)))


# --- CREATING IS_RAINING HERE TO AVOID DATA LEAKAGE ISSUE---
# I am creating this feature  directly from numeric precipitation to avoid NA string errors from codesum.
# If preciptotal > 0, it's a rainy day (1), otherwise (0).
joined_tables$is_raining <- ifelse(is.na(joined_tables$preciptotal), 0, 
                                   ifelse(joined_tables$preciptotal > 0, 1, 0))


# --- EXTRACTING  FEATURES ---
#Here I will be changing each date into a number from 1-7 with 1 being monday, 2 being tuesday and so on.
#Then I'm using a conditional to categorise whether the date is a weekend or not.
#This is done by computing whether the day number is a 6(saturday) or 7(sunday), if it is then the row is marked as 1 for is_weekend and 0 if it is not.
joined_tables$weekday_number <- wday(joined_tables$date, week_start = 1)
joined_tables$is_weekend <- ifelse(joined_tables$weekday_number %in% c(6, 7), 1, 0)
joined_tables$month_number <- month(joined_tables$date)

# month_number does not follow a linear pattern (12=December but 1=January), so I will be using case_when to categorise the months into seasons
joined_tables$season <- case_when(
  joined_tables$month_number %in% c(12, 1, 2) ~ "Winter",
  joined_tables$month_number %in% c(3, 4, 5) ~ "Spring",
  joined_tables$month_number %in% c(6, 7, 8) ~ "Summer",
  joined_tables$month_number %in% c(9, 10, 11) ~ "Autumn"
)
joined_tables$season <- as.factor(joined_tables$season)

# --- DROPPING OUTLIERS & LOG TRANSFORM ---
# The summary feature on Rstudio show that units_sold is heavily skewed. I will need to apply a log transformation to deal with the 0s and the high variance.
joined_tables$units_log <- log(joined_tables$units + 1)

#JUSTIFICATION OF LOG
#Since units_sold is zero_inflated and right skewed I have applied a log(units+1)transformation, this compresses the high value outlies and keeps the days with 0 sales still meaningful.
#This also allows me to keep all rows
#This is a better choice compared to using the IQR which would be distorted with the large amount of 0 sales days.


# --- FEATURE ENGINEERING NOTE ---
# Note on Seasonality: The 'season' variable is engineered exclusively for Section C visualisations.
#I have intentionally decided to exclude it from the predictive models because 'tavg' captures continuous temperature variances at a better resolution.

# --- 3-WAY SPLIT & MEDIAN IMPUTATION ---
#I have decided to replace the NAs with the median instead of the mean as the median is not influenced by outliers.
#To ensure that the test set does not leak into our median I will need to create a train validation test split(3 way split)
#This will be a 60/20/20 split.
#60% of the data is used to learn patterns
#20% of the data is used as a practice test
#The final 20% is used to prove that the model works on unseen data

set.seed(123)
spec = c(train = .6, validate = .2, test = .2)
g = sample(cut(seq_len(nrow(joined_tables)), nrow(joined_tables) * cumsum(c(0, spec)), labels = names(spec)))
res = split(joined_tables, g)

train_set <- res$train
val_set   <- res$validate
test_set  <- res$test

# Now computing the median statistics ONLY from the training split to ensure no data leakage takes place
columns_to_impute <- c("tmax", "tmin", "tavg", "depart", "dewpoint", "wetbulb", 
                       "heat", "cool", "sunrise", "sunset", "snowfall", 
                       "preciptotal", "stnpressure", "sealevel", "resultspeed", 
                       "resultdir", "avgspeed")

train_medians <- sapply(train_set[columns_to_impute], median, na.rm = TRUE)

# Applying those training medians across all datasets 
for(col in columns_to_impute) {
  train_set[[col]][is.na(train_set[[col]])] <- train_medians[col]
  val_set[[col]][is.na(val_set[[col]])]     <- train_medians[col]
  test_set[[col]][is.na(test_set[[col]])]    <- train_medians[col]
}


#creating boxplots with log data
#This is to visually inspect the spread of my data and see potential outliers
#Log transformation is helpful here as it squashes the impact of extreme outliers and accomodates the days with 0 sales.
units_bxplot <- ggplot(train_set, aes(x = "", y = units)) +
  geom_boxplot() +
  labs(title = "Boxplot of Raw Units Sold", x = "", y = "Units")

units_log_bxplot <- ggplot(train_set, aes(x = "", y = units_log)) +
  geom_boxplot() +
  labs(title = "Boxplot of Log Transformed Units", x = "", y = "Log(Units + 1)")

units_bxplot
units_log_bxplot
#You can visibly see difference between a boxplot with raw units and log units.

# ==========================================
# PREDICTIVE MODELLING - LINEAR REGRESSION
# ==========================================

#I need to ensure that data pooling does not take place so I am subsetting the training and validation data by item number

# --- Product 5 Data Splits ---
train_p5 <- subset(train_set, item_nbr == 5)
val_p5   <- subset(val_set, item_nbr == 5)

# --- Product 9 Data Splits ---
train_p9 <- subset(train_set, item_nbr == 9)
val_p9   <- subset(val_set, item_nbr == 9)

# --- Product 45 Data Splits ---
train_p45 <- subset(train_set, item_nbr == 45)
val_p45   <- subset(val_set, item_nbr == 45)

#Now that I have the train and validation sets for each top product I can fit the linear regression model
#I will make sure to include tavg, is_weekend and is_raining as predictors
lm_model_p5  <- lm(units_log ~ tavg + is_weekend + is_raining, data = train_p5)
lm_model_p9  <- lm(units_log ~ tavg + is_weekend + is_raining, data = train_p9)
lm_model_p45 <- lm(units_log ~ tavg + is_weekend + is_raining, data = train_p45)

#Checking the model summaries
summary(lm_model_p5)
summary(lm_model_p9)
summary(lm_model_p45)

# ==========================================
# PREDICTIVE MODELLING - REGRESSION TREES
# ==========================================

# loading in necessary packages
library(rpart)
library(rpart.plot)

# Setting the seed here guarantees that the tree-building algorithm reproduces the exact same splits when my tutor runs the script.
set.seed(123)

# --- Fitting Decision Trees per Product ---
# I will be using the same coefficients as last time
# Lowering the complexity parameter (cp) to force splits on weaker signals
tree_model_p5  <- rpart(units_log ~ tavg + is_weekend + is_raining, data = train_p5, method = "anova", control = rpart.control(cp = 0.001))
tree_model_p9  <- rpart(units_log ~ tavg + is_weekend + is_raining, data = train_p9, method = "anova", control = rpart.control(cp = 0.001))
tree_model_p45 <- rpart(units_log ~ tavg + is_weekend + is_raining, data = train_p45, method = "anova", control = rpart.control(cp = 0.001))

# --- Visualizing the Decision Trees ---
# This ticks the requirement to include a visualization of the tree
rpart.plot(tree_model_p5, main = "Decision Tree for Product 5 Sales")
rpart.plot(tree_model_p9, main = "Decision Tree for Product 9 Sales")
rpart.plot(tree_model_p45, main = "Decision Tree for Product 45 Sales")


# ==========================================
# MODEL EVALUATION ON VALIDATION DATA
# ==========================================

# calculating R-squared on validation data
get_val_r2 <- function(model, val_data, actual_val) {
  preds <- predict(model, newdata = val_data)
  rss <- sum((actual_val - preds) ^ 2)
  tss <- sum((actual_val - mean(actual_val)) ^ 2)
  return(1 - (rss / tss))
}

# --- Linear Regression Validation R2 ---
lm_r2_p5  <- get_val_r2(lm_model_p5, val_p5, val_p5$units_log)
lm_r2_p9  <- get_val_r2(lm_model_p9, val_p9, val_p9$units_log)
lm_r2_p45 <- get_val_r2(lm_model_p45, val_p45, val_p45$units_log)

# --- Decision Tree Validation R2 ---
tree_r2_p5  <- get_val_r2(tree_model_p5, val_p5, val_p5$units_log)
tree_r2_p9  <- get_val_r2(tree_model_p9, val_p9, val_p9$units_log)
tree_r2_p45 <- get_val_r2(tree_model_p45, val_p45, val_p45$units_log)

# summary table to compare them
performance_comparison <- data.frame(
  Product = c("Product 5", "Product 9", "Product 45"),
  Linear_Regression_R2 = c(lm_r2_p5, lm_r2_p9, lm_r2_p45),
  Decision_Tree_R2 = c(tree_r2_p5, tree_r2_p9, tree_r2_p45)
)
print(performance_comparison)


# ==========================================
# SECTION C: INTERPRETATION & VISUALISATION
# ==========================================
library(ggplot2)
library(dplyr)

# --- 1. Seasonal Comparison Bar Chart ---
# Combining the datasets to show average sales by season
seasonal_summary <- train_set %>%
  group_by(item_nbr, season) %>%
  summarize(mean_sales = mean(units_log, na.rm = TRUE), .groups = 'drop')

ggplot(seasonal_summary, aes(x = season, y = mean_sales, fill = as.factor(item_nbr))) +
  geom_bar(stat = "identity", position = "dodge") +
  scale_fill_brewer(palette = "Set2", name = "Product ID") +
  labs(
    title = "Plot 1: Average Log Sales Across Seasons",
    x = "Season",
    y = "Average Log Units Sold"
  ) +
  theme_minimal()

# --- 2. Rain Impact Comparison Bar Chart ---
# Showing how a rainy day shifts the mean log sales for each product
rain_summary <- train_set %>%
  group_by(item_nbr, is_raining) %>%
  summarize(mean_sales = mean(units_log, na.rm = TRUE), .groups = 'drop')

ggplot(rain_summary, aes(x = as.factor(is_raining), y = mean_sales, fill = as.factor(item_nbr))) +
  geom_bar(stat = "identity", position = "dodge") +
  scale_x_discrete(labels = c("Clear Day", "Rainy Day")) +
  scale_fill_brewer(palette = "Set1", name = "Product ID") +
  labs(
    title = "Plot 2: Impact of Rain on Average Log Sales",
    x = "Weather Condition",
    y = "Average Log Units Sold"
  ) +
  theme_minimal()

# --- 3. Product 45: Cold Weather Target Plot ---
# Visualizing the continuous temperature data alongside the tree split
ggplot(train_p45, aes(x = tavg, y = units_log)) +
  geom_point(alpha = 0.3, color = "darkgreen") +
  geom_vline(xintercept = 51, linetype = "dashed", color = "darkred", size = 1) +
  labs(
    title = "Plot 3: Product 45 Cold-Weather Demand Spike below 51°F",
    x = "Average Temperature (°F)",
    y = "Log Units Sold"
  ) +
  annotate("text", x = 46, y = 5.5, label = "Cold", color = "darkred", fontface = "bold") +
  annotate("text", x = 56, y = 5.5, label = "Warm", color = "darkred", fontface = "bold") +
  theme_minimal()