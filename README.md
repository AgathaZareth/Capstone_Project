# <center>Drafting New Talent for SF Giants 2023 Season</center>
## Linear Regression of MLB teams' percentage of wins from the last 5 regular seasons



![close-up of a worn baseball on a lush green field](https://camo.githubusercontent.com/6da502f8786aa1102f2fce163902cec9f2b2405891e79e55e2db111412af56eb/68747470733a2f2f7777772e616c756d6e692e637265696768746f6e2e6564752f732f313235302f696d616765732f656469746f722f616c756d6e695f736974652f656d61696c732f6261736562616c6c5f696e5f67726173735f636f7665725f322e6a7067)


# 1. Introduction
Intuitively, it makes sense that the performance of the team as a whole is more important than individual players themselves, we are all familiar with the idiom "greater than the sum of its parts" and baseball teams are no exception. This notebook will provide an understanding of how a teams cumulative statistics influence the percentage of wins in their regular season games. With this inferentail understanding there is also predictive capabilities, that is to say, the be able to take in the statistics of a team, then to *predict* that teams win percentage for their regular season. The effectiveness of this predictive model will be measured by how well it predicts win percentages in a test set; a set that I have the answers for but the model does not. 

The insight provided by the inferential aspects will guide my recruitment recommendations and the predictive ability will test the new rosters potential win percentage. 


# 2. Business Understanding
<p>San Francisco Giants had an unremarkable 2022 season. This year SF Giants General Manager (Pete Putila), SF Giants Senior Director of Player Development (Kyle Haines), and Senior Director of Amatuer Scouting (Micheal Holmes) are looking to invest a huge portion of their efforts into recruiting from college and minor league levels. Beyond looking at an individual player's potential, they want predictions on the collective cohesiveness of a team and how the team as a whole will perform throughout the season. <img src=https://i.pinimg.com/736x/0e/68/ed/0e68eda6243faa5f754b1cfb2b04846d--giants-sf-giants-baseball.jpg width="125", alt="SF Giants logo" style="float:left; vertical-align:middle;margin: 200px - 100px">The most obvious metric to evaluate this is a teams percentage of wins during a regular season. </p>


# 3. Overview of additional notebooks contained in this repository

## 3 - A. Sourcing Data
I have sourced all my own data and did not use any premade datasets. 
All the data collected came from web scraping of various websites. Each set needed quite a bit of code to acquire, and then clean, so this resulted in several notebooks. To better understand my process I have a brief overview of what each notebook contains, below. Each of the below dataframes created from webscraping have been pickled and saved in this repository. 

### 3 A - a. Web Scraping Player Stats

#### 3 A a- i.  Division I Collegiate Player Stats
can be found in the [College_table notebook](https://github.com/AgathaZareth/Capstone_Project/blob/main/notebooks/College_table.ipynb). Here I got a list of Division I colleges from [TheBaseballCube.com](https://thebaseballcube.com). Then using this list I was able to select only division 1 college slugs from [D1Baseball.com](https://d1baseball.com). From there I was able to scrap player hitting stats for 2022 from each division 1 college. 

#### 3 A a - ii.  Triple-A Minor League Player Stats
can be found in the [MiLB_table notebook](https://github.com/AgathaZareth/Capstone_Project/blob/main/notebooks/MiLB_table.ipynb). From [MiLB.com](https://www.milb.com) I first got a list of team id numbers for the triple A teams, then I used those to change url slugs to get player hitting stats. Fortunately, this website defaults to showing qualified players only so the resulting data frame is much smaller because it is already filtered to just the relevant players. 

#### 3 A a - iii.   MLB Player Stats
can be found in the [MLB_5_seasons notebook](https://github.com/AgathaZareth/Capstone_Project/blob/main/notebooks/MLB_5_seasons.ipynb). From [MLB.com](https://www.mlb.com)  I swapped out years and page numbers in url's to get players hitting stats for 5 seasons. 

#### <center>The above 3 mentioned player stats DF's contain the following data</center> 

| Column     | Description   |
|------------|:--------------|
| `Team`                  | **Team abbreviation** (or school name) |
| `Games Played`          | **Games in which a player has appeared.**  |
| `At Bats`               | **Trips to the plate that do not result in a walk, hit by pitch, sacrifice, or reach on interference.**  |
| `Runs`                  | **When a baserunner safely reaches home plate and scores.**  |
| `Hits`                  | **When a batter reaches base safely on a fair ball unless the batter is deemed by the official scorer to have reached on an error or a fielder's choice.**  |
| `Doubles`               | **When a batter reaches on a hit and stops at second base or only advances farther than second base on an error or a fielder's attempt to put out another baserunner.**  |
| `Triples`               | **When a batter reaches on a hit and stops at third base or only advances farther than third base on an error or a fielder's attempt to put out another baserunner.**  |
| `Home Runs`             | **When a batter reaches on a hit, touches all bases, and scores a run without a putout recorded or the benefit of error.**  |
| `Runs Batted In`        | **Runs which score because of the batter's safe hit, sac bunt, sac fly, infield out or fielder's choice or is forced to score by a bases loaded walk, hit batter, or interference.**  |
| `Walks`                 | **When a batter is awarded first base after four balls have been called by the umpire or the opposing team opts to intentionally award the batter first base.**  |
| `Strikeouts`            | **When the umpire calls three strikes on the batter.**  |
| `Stolen Bases`          | **When the runner advances one base unaided by a hit, a putout, an error, a force-out, a fielder's choice, a passed ball, a wild pitch, or a walk.**  |
| `Caught Stealing`       | **When a runner attempts to steal but is tagged out before safely attaining the next base.**  |
| `Batting Average`       | **The rate of hits per at bat against a pitcher. (formula: Hits/At Bats)**  |
| `On-Base Percentage`    | **The rate at which a batter reached base in his plate appearances. (formula: (H+BB+HBP)/(AB+BB+HBP+SF) )**  |
| `Slugging Percentage`   | **The rate of total bases per at bat. (formula: (1B+2Bx2+3Bx3+HRx4)/At Bats)**  |
| `On-Base Plus Slugging` | **The sum of on-base percentage and slugging percentage. (formula: On-Base Percentage+Slugging Percentage)**  |
| `Year`                  | **Year**  |
| `Player Name`           | **Player's name**  |
| `Position`              | **Position of player**  |


### 3 A - b.  Web Scraping Game Stats


#### 3 A b - i.  MLB Game Stats
can be found in [Games_by_day notebook](https://github.com/AgathaZareth/Capstone_Project/blob/main/notebooks/Games_by_day.ipynb). Also from [MLB.com](https://www.mlb.com), I collected data on each game of the regular seasons. I was able to create dataframes for each season.
 
| Column     | Description   |
|------------|:--------------|
| `Day`                  | **Day of the week**  |
| `Month`                | **Month Abbreviation**  |
| `Date`                 | **Date of the month**  |
| `Away`                 | **Away team**  |
| `Home`                 | **Home team**  |
| `Win`                  | **Winning team**  |
| `W Score`              | **Winning teams score**  |
| `Lose`                 | **Losing team**  |
| `L Score`              | **Losing teams score**  |
| _`Year`_               | _**each df was saved by year/season. `year` column was added later, then combined with `Month` and `Date` and converted to YYYY-MM-DD format**_  |


### 3 A - c. Creating Team Stats


#### 3 A c - i.  MLB Team Stats
can be found in the [Aggregate_team_stats](https://github.com/AgathaZareth/Capstone_Project/blob/main/notebooks/Aggregate_team_stats.ipynb) notebook. This is where I combined the MLB player and game stats into one df. First, I used the MLB player stats and filtered out players with less than 5 at bats, ie pitchers  who might drive down team averages. Next I found the cumulative totals of the players on a team, as well as their averages ((cumulative totals)/(number of players > 5 at bats)). Secondly, I used MLB game stats to get teams number of wins and losses for each season, to get the win percentage, per team, per season. Finally, I added the win percentages to the aggregated stats table. 
 
| Column     | Description   |
|------------|:--------------|
| `Team`                      | **Team abbreviation**  |
| `Year`                      | **Year/Season**  |
| `Games Played Sum`          | **Cumulative sum of games played.**  |
| `At Bats Sum`               | **Cumulative sum of trips to the plate that do not result in a walk, hit by pitch, sacrifice, or reach on interference.**  |
| `Runs Sum`                  | **Cumulative sum of when a baserunner safely reaches home plate and scores.**  |
| `Hits Sum`                  | **Cumulative sum of when a batter reaches base safely on a fair ball unless the batter is deemed by the official scorer to have reached on an error or a fielder's choice.**  |
| `Doubles Sum`               | **Cumulative sum of when a batter reaches on a hit and stops at second base or only advances farther than second base on an error or a fielder's attempt to put out another baserunner.**  |
| `Triples Sum`               | **Cumulative sum of when a batter reaches on a hit and stops at third base or only advances farther than third base on an error or a fielder's attempt to put out another baserunner.**  |
| `Home Runs Sum`             | **Cumulative sum of when a batter reaches on a hit, touches all bases, and scores a run without a putout recorded or the benefit of error.**  |
| `Runs Batted In Sum`        | **Cumulative sum of runs which score because of the batter's safe hit, sac bunt, sac fly, infield out or fielder's choice or is forced to score by a bases loaded walk, hit batter, or interference.**  |
| `Walks Sum`                 | **When a batter is awarded first base after four balls have been called by the umpire or the opposing team opts to intentionally award the batter first base.**  |
| `Strikeouts Sum`            | **Cumulative sum of when the umpire calls three strikes on the batter.**  |
| `Stolen Bases Sum`          | **Cumulative sum of when the runner advances one base unaided by a hit, a putout, an error, a force-out, a fielder's choice, a passed ball, a wild pitch, or a walk.**  |
| `Caught Stealing Sum`       | **Cumulative sum of when a runner attempts to steal but is tagged out before safely attaining the next base.**  |
| `Mean Games Played`         | **Average number of Games played.**  |
| `Mean At Bats`              | **Average number of trips to the plate that do not result in a walk, hit by pitch, sacrifice, or reach on interference**  |
| `Mean Runs`                 | **Average number of runs when a baserunner safely reaches home plate and scores .**  |
| `Mean Hits`                 | **Average number of times when a batter reaches base safely on a fair ball unless the batter is deemed by the official scorer to have reached on an error or a fielder's choice.**  |
| `Mean Doubles`              | **Average number of times when a batter reaches on a hit and stops at second base or only advances farther than second base on an error or a fielder's attempt to put out another baserunner.**  |
| `Mean Triples`              | **Average number of times when a batter reaches on a hit and stops at third base or only advances farther than third base on an error or a fielder's attempt to put out another baserunner.**  |
| `Mean Home Runs`            | **Average number of times a batter reaches on a hit, touches all bases, and scores a run without a putout recorded or the benefit of error.**  |
| `Mean Runs Batted In`       | **Average number of runs which score because of the batter's safe hit, sac bunt, sac fly, infield out or fielder's choice or is forced to score by a bases loaded walk, hit batter, or interference.**  |
| `Mean Walks`                | **Average number of times a batter is awarded first base after four balls have been called by the umpire or the opposing team opts to intentionally award the batter first base.**  |
| `Mean Strikeouts`           | **Average number of times when the umpire calls three strikes on the batter.**  |
| `Mean Stolen Bases`         | **Average number of times when the runner advances one base unaided by a hit, a putout, an error, a force-out, a fielder's choice, a passed ball, a wild pitch, or a walk.**  |
| `Mean Caught Stealing`      | **Average number of times when a runner attempts to steal but is tagged out before safely attaining the next base.**  |
| `% wins`      | **The percentage of wins of regular season games.**  |


# 4. Notebook Setup

## 4 - A. Imports
Reproducibility is an imortant consideration. So, in addition to all the created and used dataframes being pickled, I have exported my current working environment with not only the list of packages used, but also the specific versions of those packages. This file is called `environment.yml`, you can find all relevant import data [HERE](https://github.com/AgathaZareth/Capstone_Project/blob/main/environment.yml). In this section of my notebook I have also set a random state seed of 137.


## 4 - B. Functions
I like to put anything that is used more than once into a function to avoid the copy-past look of the notebook. Additionally, this keeps the flow of the notebook smoother and just generally cleaner looking. I will not list out all the functions I created here in the read me, see this section, 4-B, in the [Modeling notebook](https://github.com/AgathaZareth/Capstone_Project/blob/main/notebooks/Modeling.ipynb) if needeed. 


# 5. Data Understanding
The data comes from web scraping [MLB.com](https://www.mlb.com/stats/). I took the last 5 years of players hitting stats over regular seasons and cumulated them into team stats. I also took game details to determine team win percentages per season. To avoid collinearity issues down the line I only added statistics that did not have any direct relationship to other stats, i.e. I did not include things that already combined other stats. For example, batting average combines `hits` and `at bats` so I included hits and at bats but left out batting average. Any player stats with a formula was left out. 


## 5 - A. Load raw data
In this section I load the data frame created in the [Aggregate_team_stats](https://github.com/AgathaZareth/Capstone_Project/blob/main/notebooks/Aggregate_team_stats.ipynb) notebook. See above section `3Ac-i. MLB Team Stats`, for list of column names and descriptions.


# 6. Exploratory Data Analysis of Raw Data
It is important to look at data before making any assumptions. I think linear regression is a good modeling option based on the problem but it would be erroneous to move forward without some type of data check. The goal is to establish if there is a linear relationship between the different statistics and a teams win percentage. I also need to check if there are any correlations between the independent variables themselves. Exploratory data analysis can also help identify outliers and determine weather appropriate to remove or leave as they are. 


## 6 - A. Drop unnecessary columns
I do not want the model looking for trends in the `Year` or `Team` features so these need to be dropped. 


## 6 - B.  Identify target feature -  `% wins`
This is just for convenience as I move through the notebook.


## 6 - C. EDA basics


### 6 C - a. Check shape
Here I can see my data has 150 rows (datapoints) and 25 columns (24 independent variables and target variable)

### 6 C - b. Check for nulls
There are zero null or missing values in this df.

### 6 C - c. Check info
pandas.DataFrame.info method prints information about a DataFrame including the index dtype and columns, non-null values and memory usage.
![raw data dot info printout](../images/ss_examples/raw_df.info.png)

<img src="/Users/me/Development/FlatironProjects/phase_5/Capstone_Project/images/ss_examples/raw_df.info.png" alt="raw data dot info printout" />

<div class="alert alert-block alert-info">
<b>'Check info' notes:</b> There are no missing values, however, the above shows all the independent variables are strings; they need to be converted to numeric values. </div>


#### 6 C  c - i. Convert Dtypes
I do a blanket conversion of the entire df since all features, independent and dependent, need to be float64.
<img src="../images/ss_examples/converted_to_floats_info.png" alt="raw data converted to floats dot info printout" />

<div class="alert alert-block alert-success">
<b>Success:</b> All Dtypes are now floats.
</div>

### 6 C - d. Distribution of values
An important consideration when using multiple predictors in any machine learning model is the scale of these features. 

    
    
</body>
</html>