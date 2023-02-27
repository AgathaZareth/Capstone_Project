<h1 align="center" width="100%">
Drafting New Talent for SF Giants 2023 Season
</h1>

<h2>Linear Regression of MLB teams' percentage of wins from the last 5 regular seasons</h2>



![close-up of a worn baseball on a lush green field](https://camo.githubusercontent.com/6da502f8786aa1102f2fce163902cec9f2b2405891e79e55e2db111412af56eb/68747470733a2f2f7777772e616c756d6e692e637265696768746f6e2e6564752f732f313235302f696d616765732f656469746f722f616c756d6e695f736974652f656d61696c732f6261736562616c6c5f696e5f67726173735f636f7665725f322e6a7067)


# 1. Introduction

Intuitively, it makes sense that the performance of the team as a whole is more important than individual players themselves, we are all familiar with the idiom "greater than the sum of its parts" and baseball teams are no exception. This notebook will provide an understanding of how a teams cumulative statistics influence the percentage of wins in their regular season games. With this inferentail understanding there is also predictive capabilities, that is to say, the be able to take in the statistics of a team, then to *predict* that teams win percentage for their regular season. The effectiveness of this predictive model will be measured by how well it predicts win percentages in a test set; a set that I have the answers for but the model does not. 

The insight provided by the inferential aspects will guide my recruitment recommendations and the predictive ability will test the new rosters potential win percentage. 


# 2. Business Understanding

<p align="left" width="100%"><img align="left" width="22%" src=https://i.pinimg.com/736x/0e/68/ed/0e68eda6243faa5f754b1cfb2b04846d--giants-sf-giants-baseball.jpg width="125", alt="SF Giants logo">
San Francisco Giants had an unremarkable 2022 season. This year SF Giants General Manager (Pete Putila), SF Giants Senior Director of Player Development (Kyle Haines), and Senior Director of Amatuer Scouting (Micheal Holmes) are looking to invest a huge portion of their efforts into recruiting from college and minor league levels. Beyond looking at an individual player's potential, they want predictions on the collective cohesiveness of a team and how the team as a whole will perform throughout the season. The most obvious metric to evaluate this is a teams percentage of wins during a regular season. 

</p>


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

<p align="center" width="100%">The above 3 mentioned player stats DF's contain the following data</p> 

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

<p align="center" width="100%">
<img src="/images/ss_examples/raw_df_info.png" alt="raw data dot info printout">
    
<img src="/images/ss_examples/check_info_notes.png" alt="blue note box: There are no missing values, however, the above shows all the independent variables are strings; they need to be converted to numeric values">    
</p>



#### 6 C  c - i. Convert Dtypes

I do a blanket conversion of the entire df since all features, independent and dependent, need to be float64.

<p align="center" width="100%">
<img src="/images/ss_examples/converted_to_floats_info.png" alt="raw data converted to floats dot info printout">
    
<img src="/images/ss_examples/success_dtypes_floats.png" alt="green success box: All Dtypes are now floats.">    
</p>



### 6 C - d. Distribution of values
An important consideration when using multiple predictors in any machine learning model is the scale of these features. 


#### 6 C d - i. Check describe

I will use pandas.DataFrame.describe to generate descriptive statistics. Descriptive statistics include those that summarize the central tendency, dispersion and shape of a dataset’s distribution, excluding NaN values.

- The count of not-empty values
- The average (mean) value
- The standard deviation
- the minimum value
- The 25% percentile
- The 50% percentile
- The 75% percentile
- the maximum value

I will transpose it for easier viewing.

<p align="center" width="100%">
<img src="/images/ss_examples/raw_describe.png" alt="raw data dot describe">

<img src="/images/ss_examples/check_describe_notes.png" alt="blue note box: A quick scroll down the mean, min, & max columns I can see there is a huge range in each of the independent features. Variables of vastly different scales can impact the influence over the model. To avoid this, it is best practice to normalize the scale of all features before feeding the data into a machine learning algorithm. I will need to standardize my data so the features with larger numeric values are not unfairly weighted by the model. To avoid any potential data leakage, I will first split the data before altering it in any way.">

</p>



#### 6 C d - ii. Plot distributions of each feature

When deciding which method to use when scaling, it can be helpful to understand the distribution of values so I want to do a quick histogram plot of each feature. I will use seaborn.histplot for this, documentation [HERE](https://seaborn.pydata.org/generated/seaborn.histplot.html). Presumably, the split data will have similar distributions. If I wanted to be extremely careful I could view distributions AFTER I split but in this particular case I think it is fine to view now. `hist_grid` function used below.

<p align="center" width="100%">
<img src="/images/ss_examples/raw_data_distribution_of_values.png" alt="histogram thumbnail tiles of each feature">

<img src="/images/ss_examples/blue_note_box_raw_histo_tiles.png" alt="blue note box: Most features have a roughly normal distribution. I don't see any extreme skewness that might justify loggin a variable.">

</p>



# 7. Preprocessing


## 7 - A. Train Test Split

Use `sklearn.model_selection.train_test_split` (documentation [HERE](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)) to create a train and test set. I will withhold 20% of data from the models learning-data, then use that 20% to test and evaluate my models performance.

### 7 A - a. Separate data into features and target

### 7 A - b. Split data into train and test sets
```

X_train has a shape of: (120, 24)
y_train has a shape of: (120,)

X_test has a shape of: (30, 24)
y_test has a shape of: (30,)

```

## 7 - B. Scale Data

Feature scaling is a method used to normalize the range of the independent variables of data. A machine learning algorithm can only see numbers; this means, if there is a vast difference in the feature ranges (as there is with this data, demonstrated in step 4f) it makes the underlying assumption that higher ranging numbers have superiority of some sort and these more significant numbers start playing a more decisive role while training the model. Therefore, feature scaling is needed to bring every feature on the same footing.

**Standardization**

Feature standardization makes the values of each feature in the data have zero mean and unit variance. The general method of calculation is to determine the distribution mean and standard deviation for each feature and calculate the new data point by the following formula:

$$x' = \dfrac{x - \bar x}{\sigma}$$

x' will have mean $\mu = 0$ and $\sigma = 1$

Note that standardization does not make data $more$ normal, it will just changes the mean and the standard error!

**Normalization**
- Min-max scaling
 - This way of scaling brings all values between 0 and 1. 
 
$$x' = \dfrac{x - \min(x)}{\max(x)-\min(x)}$$


- Mean normalization
 - The distribution will have values between -1 and 1, and a mean of 0.
 
$$x' = \dfrac{x - \text{mean}(x)}{\max(x)-\min(x)}$$

- You can bound your normalization range by any interval `[a,b]` with

$$x' = a + \dfrac{(x - \min(x))(b - a)}{\max(x)-\min(x)}$$


---

Choosing which method to use depends on the distribution of your data.  A couple of relevant generalizations are: 

- Standardization may be used when data represent Gaussian Distribution, while Normalization is great with Non-Gaussian Distribution
- Impact of Outliers is very high in Normalization



### 7 B - a. Scale and create new scaled dfs

Because my data is normally distributed I will use `sklearn.preprocessing.StandardScaler` default standardization (documentation [HERE](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)):

$$z = (x - u) / s$$

Where u is the mean of the training samples, and s is the standard deviation of the training samples. 


### 7 B - b. Explore the scaling effect on training data

#### 7 B b - i. Check `X_train_scaled` describe


<p align="center" width="100%">
<img src="/images/ss_examples/scaled_describe.png" alt="dot describe df of scaled data">

<img src="/images/ss_examples/scaled_describe_notes.png" alt="blue note box: Now that the df has been scaled this .describe() method is less useful. A visualization will explain the transformation better.">

</p>



#### 7 B b - ii. Visualizations
pandas.DataFrame.describe method is great but it fails to truely convey how the data is transformed by scaling. The best way to view this change is with boxplots and histograms. 

##### 7 B b ii - 1. boxplots
seaborn.boxplot show the minimum, first quartile, median, third quartile, and maximum of features, documentation [HERE](https://seaborn.pydata.org/generated/seaborn.boxplot.html).
Boxplots are a great way to see the change in the range of each independent feature before and after standard scaling. `boxplots` function used below.


<p align="center" width="100%">
<img src="/images/ss_examples/boxplot_before_scaling.png" alt="boxplot of data BEFORE scaling">

<img src="/images/ss_examples/boxplot_after_scaling.png" alt="boxplot of data AFTER scaling">

</p>

##### 7 B b ii - 2. histograms
An additional way to view the scaling effect is through histograms. If you think of boxplots as a top view of distributions, then you can think of histograms as a side view. Imagine yourself standing on the right hand side of the above boxplots looking down the 0 line. `hist_overlay` function used below.

<p align="center" width="100%">
<img src="/images/ss_examples/histos_before_scaling.png" alt="histograms of data BEFORE scaling">

<img src="/images/ss_examples/histos_after_scaling.png" alt="histograms of data AFTER scaling">

<img src="/images/ss_examples/vizualization_notes.png" alt="blue note box: You can see the means of all the independent variables are roughly around 0 and they all have roughly the same min and max values. From the boxplots, you can also see there are a few outliers, primarily in the team averaged variables.">
    
</p>

## 7 - C. Feature Reduction

Before moving on I want to regplot all the independent variables. The first 12 features are cumulative totals of team stats and the second 12 are those same stats but divided by the number of players to get the mean. If I use both, the cumulative totals and the averages, there will be a lot of multicollinearity. Multicollinearity occurs when two or more independent variables are highly correlated with one another, and as you can imagine using one statistic to acquire the other would have a high correlation. So I need to make a choice on which set to use. 


### 7 C - a. Regplots
Use regplots to show me which features have a 'cleaner' linear relationship with target variable. This method is used to plot data and a linear regression model fit. There are a number of mutually exclusive options for estimating the regression model see documentation [HERE](https://seaborn.pydata.org/generated/seaborn.regplot.html). `regplot_grid` function used below.

<p align="center" width="100%">
<img src="/images/ss_examples/regplot_all_feats.png" alt="regplots of all 24 independent variables">

<img src="/images/ss_examples/regplot_notes.png" alt="blue note box: It appears to me that the cumulative sum totals have either the same or even cleaner linear relationships with the target variable. You can see this with just the scatter plots alone but the red shaded band is perhaps a better representation. This red line shows what a linear model might look like, with the shaded part being the confidence interval. The more 'noise' the wider that red shaded area. A perfectly confident model would be a clean red line without a shaded band. I will use heatmaps to investigate further.">
    
</p>

### 7 C - b. Heatmaps

The regplots show the relationship between the features and target variable. The heatmap will show the  correlations between all of the numeric values (in this case, all the features) in our data. The x and y axis labels indicate the pair of values that are being compared, and the color and the number are both representing the correlation. Color is used here to make it easier to find the largest/smallest numbers. The bottom row is particularly important because it shows all the features correlation with the target variable but I am also looking for strong correlation between the independent variables. Documentation [HERE](https://seaborn.pydata.org/generated/seaborn.heatmap.html). `heatmap` function used below.

#### 7 C b - i. Set up for heatmap plots
Converting `train_scaled` and `test_scaled` to a pandas df reset the indices. I now need to convert `y_train` and `y_test` to a pandas df and reset the indices so I can concat `train_scaled` and `y_train` to create a visualization df. 


##### 7 C b i - 1. Heatmap - cumulative features

<p align="center" width="100%">
<img src="/images/ss_examples/heatmap_cumulative.png" alt="heatmap of the 12 cumulative independent variables and the target variable">
    
</p>

##### 7 C b i - 2. Heatmap - averaged features

<p align="center" width="100%">
<img src="/images/ss_examples/heatmap_averaged.png" alt="heatmap of the 12 averaged independent variables and the target variable">
    
<img src="/images/ss_examples/heatmap_notes.png" alt="blue note box: Interestingly, the averaged variables have a few features with a stronger correlation with the target variable than those of the corresponding cumulative variables. Overall however, the cumulative features seem to have stronger correlations. In addition, there are a lot more .9 (or above) inter-feature correlations among the averaged variables. The averaged variables have 18 inter correlated pairs of .9 or greater, while the cumulative variables have only 5.">
    
</p>

### 7 C - c. Selecting Features to Drop

Based on the Heatmap and Regplots I will *keep* the cumulative sum features. I need to eliminate the highly correlated features while keeping as much data as possible. This means I need to remove the least amount of features possible. `Runs Sum` has the highest correlation with `% wins` so I want to keep this feature. 


<p align="center" width="100%">
<img src="/images/elimination_tables.png" alt="table showing elimination process">
    
</p>

By eliminating `Hits Sum` & `Runs Batted In Sum` I can remove all the pairs with a correlation of .9 or greater. This leaves:
- `Games Played Sum`
- `At Bats Sum`
- `Runs Sum`
- `Doubles Sum`
- `Triples Sum`
- `Home Runs Sum`
- `Walks Sum`
- `Strikeouts Sum`
- `Stolen Bases Sum`
- `Caught Stealing Sum`


**Note: this is likely not a comprehensive selection of features to be eliminated.* 


### 7 C - d. Drop Features
Create reduced dfs from `X_test_scaled` & `X_train_scaled`

New variables 
- `X_train_s_r`: "X_train, scaled, and reduced"
- `X_test_s_r`: "X_test, scaled, and reduced"

#### 7 C d - i. Pairplot
The final visualization to check inter feature correlation is a pairplot. Pairplots are great ways to see the relationship between two features using scatter plots. Pariplots also show the distribution of each feature across the diagonal. This has already  been plotted but here we can see just the features I have decided to keep.

<p align="center" width="100%">
<img src="/images/ss_examples/pairplot.png" alt="pairplot of remaining independent variables and target feature">
    
<img src="/images/ss_examples/pairplot_notes.png" alt="blue note box: I can see there are still quite a bit of features that are correlated with eachother. These will likely be filtered out by p value when modeling.">
    
</p>

## 7 - D. Investigate Outliers

Removing outliers can reduce the errors (or residuals) of a model. However, not all outliers should be removed, and Jim, from *Statistics By Jim*, perfectly states in [Guidelines for Removing and Handling Outliers in Data](https://statisticsbyjim.com/basics/remove-outliers/), "It’s bad practice to remove data points simply to produce a better fitting model or statistically significant results." 
There are other ways to deal with outliers than to do a blanketed removal. The most common is logging. Before deciding to add more complexity to my model by logging features, I need to investigate where these outliers are, and how many? Is the complexity worth it for a few data points? `boxplots` function used below.


<p align="center" width="100%">
<img src="/images/ss_examples/outliers_boxplot.png" alt="boxplot of 10 remaining independent variables">
    
<img src="/images/ss_examples/notes_outlier_boxplots.png" alt="blue note box: Only 4 features have outliers and it looks like they each have only 1.">
    
</p>

I will check the number of outliers by using the `print_outliers` function. This will identify outliers from a column based on zscore. If 3 standard deviations away from mean the data point is considered an outlier. 

```

Triples Sum has 1 outlier(s)
Caught Stealing Sum has 1 outlier(s)

```


<p align="center" width="100%">    
<img src="/images/ss_examples/investigate_outlier_notes.png" alt="blue note box: These outliers are likely due to natural variation but my sample size is relatively low. Yes it covers 5 year of data but each season only has 24 teams i.e. data points. I think If I were able to increase my sample size, these outliers would fall within the typical gaussian bell curve.">
    
</p>

# 8. Linear Regression Modeling - Scaled Data

## 8 - A. Build baseline and final model

This is all wrapped up in a beautiful function that creates a baseline model by determining the highest positively correlated feature with `% wins`. Then it iteratively adds features based on the features p value. See `build_models` docstring for more info. 


<p align="center" width="100%">    
<img src="/images/ss_examples/build_models.png" alt="baseline and final models: histogram and qq plots of residuals, printout of r_squared, adjusted_r_squared, k_fold_score, and the second half of ols .summary report with p values and variable coefficients">
    
</p>


Save final model predictors.

```

Features used in final_model to predict win percentage: 

['Runs Sum', 'Strikeouts Sum', 'Walks Sum']

```

## 8 - B. Evaluation of Final Model

### 8B - a. Create `final_scaled_model_df`

<p align="center" width="100%">    
<img src="/images/ss_examples/final_scaled_model_df_ss.png" alt="top 5 and last 5 rows of final scaled model df - this is the 3 independent variables and target feature">
    
</p>

### 8B - b.Linear Regression Assumption checks

Regression is a powerful analysis however, if some of the necessary assumptions are not satisfied, regression makes biased and unreliable predictions. Below I check the following 4 assumptions:
- Independence Assumption
- Linearity Assumption
- Homoscedasticity Assumption
- Normality Assumption

I give brief explanations of these assumption checks at each header but you can also visit:
- [HERE](https://github.com/learn-co-curriculum/dsc-regression-assumptions#about-regression-assumptions) for a short summary of Linearity, Homoscedasticity, and Normality Assumptions. 

#### 8Bb - i. Independence Assumption

Because I am using this model for both inferential and predictive purposes I need to ensure that each observation is independent of the others. A violation of the independence assumption results in incorrect confidence intervals and p-values, it can essentially be thought of as a kind of double-counting in the model and it can produce estimates of the regression coefficients that are not statistically significant. 

This article ["How to detect and deal with Multicollinearity"](https://towardsdatascience.com/how-to-detect-and-deal-with-multicollinearity-9e02b18695f1) does a really good job of explaining the differences between using correlations vs VIF, 
>A correlation plot can be used to identify the correlation or bivariate relationship between two independent variables whereas VIF is used to identify the correlation of one independent variable with a group of other variables. Hence, it is preferred to use VIF for better understanding.
>
>- VIF = 1 → No correlation
>- VIF = 1 to 5 → Moderate correlation
>- VIF >10 → High correlation

Below I check the correlation between the 3 independent features that made it into the model. I do this using the function `collinearity_pairs` which returns pandas.DataFrame of pairs of features with correlations between .75 and 1. 

<p align="center" width="100%">
<img src="/images/ss_examples/corr_pairs.png" alt="pairs of independent variables with more than .75 correlation.">
    
<img src="/images/ss_examples/collinearity_pairs_notes.png" alt="blue note box: You can see that `Runs Sum` and `Walks Sum` have a Pearson's correlation of 0.8127. As stated above this is just a measurement of the two features with each other and not a measure of each independent variable with the group of other variables in the model. So while it is interesting to see this correlation is so high, it does not tell me much about how these two variables might impact my models inferential capabilities.">
    
</p>


Variance inflation factor is a measure of the degree of multicollinearity or correlation between the independent variables in your multiple linear regression analysis. The rules of thumb are listed above. Statsmodels has a VIF function, [statsmodels.stats.outliers_influence.variance_inflation_factor](https://www.statsmodels.org/dev/generated/statsmodels.stats.outliers_influence.variance_inflation_factor.html), and states, 
>One recommendation is that if VIF is greater than 5, then the explanatory variable given by exog_idx is highly collinear with the other explanatory variables, and the parameter estimates will have large standard errors because of this.

Based on this and the above rule of thumbs, I have a function `get_VIFs_above5` that does exactly as it sounds, it returns any feature with an VIF above 5.


<p align="center" width="100%">

<img src="/images/ss_examples/green_VIF_notes.png" alt="green success box: Nothing returned, this means there are no features that have a VIF above 5. The assumption of independence is satisfied">
    
</p>


#### 8Bb - ii. Linearity Assumption

The linearity assumption requires that there is a linear relationship between the response variable (Y) and predictor (X). Linear means that the change in Y by 1-unit change in X, is constant. If you were to try to fit a linear model to a non-linear data set, OLS would fail to capture the trend mathematically, resulting in an inaccurate relationship. This will also result in erroneous predictions on an unseen data set.

<p align="center" width="100%">

<img src="/images/ss_examples/regplots_3_feats.png" alt="regplots of three features that made it into final model.">
    
</p>



#### 8Bb - iii. Homoscedasticity Assumption

Homoscedasticity indicates that a dependent variable's variability is equal across values of the independent variable. A scatter plot is a good way to check whether the data are homoscedastic (meaning the residuals are equal across the regression line). 

<p align="center" width="100%">

<img src="/images/ss_examples/residuals_vs_predicted.png" alt="scatter plot of residuals vs predicted y values.">
    
</p>

#### 8Bb - iv. Normality Assumption

The normality assumption states that the model residuals should follow a normal distribution. This can be viewed with either a histogram or a QQ Plot of the residuals. I prefer to use both to fully understand the distribution. 


<p align="center" width="100%">

<img src="/images/ss_examples/histo_qq_resids.png" alt="histogram and qq plot of residuals.">
    
</p>


### 8B - c. Root Mean Squared Error
Root mean square error is one of the most commonly used measures for evaluating the quality of predictions. It shows how far predictions fall from measured true values using Euclidean distance (i.e. the length of a line segment between the two points).

```

MSE :  33.78809248898828
RMSE :  5.812752574210284

```

<p align="center" width="100%">

<img src="/images/ss_examples/RMSE_notes.png" alt="blue note box: This RMSE means, on average, this model is off by about 5.81%.">
    
</p>


## 8 - C. Interpreting model results
`coeffs_to_df` function used below. This function has a parameter `predictors_scaled` when set to `True` it takes the standard deviation of each predictor and names the columns in a translatable way that makes it easy to interpret the coefficients. 

<p align="center" width="100%">

<img src="/images/ss_examples/coeffs_to_df.png" alt="screen shot of coeffs_to_df, gives coefficients of independent variables from the model and how they increase or decrease percent of wins in a regular season.">
 
<img src="/images/ss_examples/interpreting_notes.png" alt="blue note box: This df shows, RUNS is the most significant feature when predicting a teams win percentage. Each increase of 118 RUNS, results, on average, in an increase of 5.78% of WINS in a teams regular season. Next is WALKS. Each increase of 153 WALKS, results, on average, in an increase of 2.43% of WINS in a team's regular season. And finally, STRIKEOUTS. Each increase of 93 STRIKEOUTS, results, on average, in a decrease of 1.83% of WINS in a teams regular season..">
    
</p>


To put this into perspective, I will create a dataframe with world series winners for 2022, 2021, 2019, 2018, and 2017. I will include just the three predictive features from the model and the teams regular season win percentage.  I will also include the SF Giants for each respective year for comparison. 

<img src="/images/ss_examples/wolrd_series_winners_and_giants.png" alt="df with 5 seasons of world series winners regular season win percentages and the sf giants win percentage for those same 5 seasons">
 
<img src="/images/ss_examples/interpreting_notes.png" alt="blue note box: The grey rows are the World Series winners, the white rows are the SF Giants team stats. You can see, with the exception of 2021, the world series winners had regular season win percentages of 10 - 22 percent more than the SF Giants. 2021 the Giants were 1st in National League West, eventually losing the National League Division Series (3-2) to the Dodgers who had a regular season win % of 65.43, less than 1% less than SF Giants.">
    
</p>


## 8 - D. Using the model for predictions

### 8D - a. Check SF Giants 2022 regular season prediction
Check the models prediction on SF Giants 22 regular season







































































































