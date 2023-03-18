<h1 align="center" width="100%">
Drafting New Talent for SF Giants 2023 Season
</h1>

<h2>Linear Regression of MLB teams' percentage of wins from the last 5 regular seasons</h2>



![close-up of a worn baseball on a lush green field](https://camo.githubusercontent.com/6da502f8786aa1102f2fce163902cec9f2b2405891e79e55e2db111412af56eb/68747470733a2f2f7777772e616c756d6e692e637265696768746f6e2e6564752f732f313235302f696d616765732f656469746f722f616c756d6e695f736974652f656d61696c732f6261736562616c6c5f696e5f67726173735f636f7665725f322e6a7067)

*This notebook is intended for educational purposes, all scenarios are hypothetical and intended to satisfy an academic assignment.*

# Introduction

We are all familiar with the idiom "greater than the sum of its parts" and baseball teams are no exception. Intuitively, it makes sense that the performance of a team as a whole is more important than individual players themselves, but what specifically makes up the magic sauce that leads to a team's winning performance?

This collection of notebooks provides an understanding of how a team's cumulative statistics influence the percentage of wins in their regular season games. I will provide 3 statistics that are the most predictive of increasing or decreasing this win percentage. By knowing which individual player stats are most indicative of team performance, we can look for players in minor and collegiate leagues who will have the greatest positive impact on the teams win percentage, and test how a roster update will alter the projected win percentage compared to the current 2023 roster. 


# Business Understanding

<p align="left" width="100%"><img align="left" width="15%" src=https://i.pinimg.com/736x/0e/68/ed/0e68eda6243faa5f754b1cfb2b04846d--giants-sf-giants-baseball.jpg width="115", alt="SF Giants logo">
San Francisco Giants had an unremarkable 2022 season. This year SF Giants executives are looking to recruit a large portion of new talent from collegiate and minor leagues. Beyond looking at an individual player's potential, they want predictions on how the team will perform throughout the season. The most obvious metric to evaluate this is a teams percentage of wins during their regular season.
</p>



# Repository Overview 

All the data collected came from web scraping of various websites. The ETL process resulted in several notebooks. These notebooks are located in the `notebooks` folder. Each notebook yielded at least 1 dataframe, which was then pickled. These saved data frames are located in the `pickled_tables` folder in this repository. Reproducibility is an important consideration so, in addition to the pickled data frames, I have exported my current working environment to a file called `environment.yml`; you can find all relevant import data in this file. To create an environment from this `environment.yml` file see conda documentation [HERE](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file).

Below is a visual overview of the flow of the notebooks and how they are utilized by the final `Modeling` notebook. For a more detailed summary of each notebook see section "3 - A. Sourcing Data" in the `Modeling` notebook.

<p align="center" width="100%">
<img src="/images/ss_examples/overview_notebooks.png" alt="overview of how additional notebooks are utilized in final modeling notebook"> 
</p>


## Breakdown of TRAIN and UTILIZE Steps

The above graphic simplifies the modeling process into **Training** and **Utilizing** the model. The below graphics offer more detail about these two steps, show what notebooks are used at different parts of these steps, provide insight as to what information is extracted from said notebooks, and how they are relevant to the final `Modeling` notebook.

<h3 align="center" width="100%">
Visual Overview of the Model Training Process
</h3>

<p align="center" width="100%">
<img src="/images/ss_examples/train_overview.png" alt="overview of utilizing trained model"> 
</p>

The above **Training** graphic ends with the trained and evaluated Model. The below graphic starts with this trained and evaluated model, then shows the steps taken to utilize the model to predict this season's win percentage and compare it to a fantasy roster's win percentage.

<h3 align="center" width="100%">
Visual Overview of the Model Utilization Process
</h3>

<p align="center" width="100%">
<img src="/images/ss_examples/overview_utilizing_model.png" alt="overview of utilizing trained model"> 
</p>



# Data Understanding

Major League Player and Game Stats - [MLB.com](https://www.mlb.com/stats/):
 - 5 regular season player hitting stats 
   -  To avoid collinearity issues down the line I only added statistics that did not have any direct relationship to other stats, i.e. I did not include things that already combined other stats. For example, `batting average` combines `hits` and `at bats` so I included `hits` and `at bats` but left out `batting average`. In summary, any player-stat with a formula was left out. 
 - Data on each game of the 5 regular seasons
   - who played who, where they played, date of game, and each teams score
 
Minor League Triple-A Player Stats - [MiLB.com](https://www.milb.com):
- 2022 player hitting stats. This website defaults to showing qualified players only, so the resulting data frame is much smaller compared to the Major and Collegiate tables. 

Collegiate Division-1 Player Stats - [TheBaseballCube.com](https://thebaseballcube.com), and [D1Baseball.com](https://d1baseball.com):
- 2022 player hitting stats



# Linear Regression Modeling - Scaled Data

## Build Baseline and Final model

The baseline model takes in the most correlated feature with `win %`. Final model was created by iteratively adding features based on the features p value.

Each model has a small printout of the following:

- histogram of distribution of residuals
- QQ plot
- r-squared & adjusted r-squared from statsmodel 
- 5 splits kfold crossvalidation r-squared 
- the second half of OLS.summary() table 
  - coefficients, 
  - std error, 
  - t, p, and ci values 


<p align="center" width="100%">    
<img src="/images/ss_examples/build_models.png" alt="baseline and final models: histogram and qq plots of residuals, printout of r_squared, adjusted_r_squared, k_fold_score, and the second half of ols .summary report with p values and variable coefficients">
    
</p> 

## Root Mean Squared Error of Test Data
Root mean square error is one of the most commonly used measures for evaluating the quality of predictions. It shows how far predictions fall from measured true values using Euclidean distance (i.e. the length of a line segment between the two points). Below are the final linear regression model's Mean Squared Error, and Root Mean Squared Error. 

```

MSE :  33.78809248898828
RMSE :  5.812752574210284

```

<p align="center" width="100%">

<img src="/images/ss_examples/RMSE_notes.png" alt="blue note box: This RMSE means, on average, this model is off by about 5.81% in it's predictions.">
    
</p>



# Interpreting Linear Regression Model Results

<p align="center" width="100%">

<img src="/images/ss_examples/coeffs_to_df.png" alt="screen shot of coeffs_to_df, gives coefficients of independent variables from the model and how they increase or decrease percent of wins in a regular season.">
 
<img src="/images/ss_examples/coeff_to_df_notes2.png" alt="blue note box: This df shows, RUNS is the most significant feature when predicting a teams win percentage. Each increase of 118 RUNS, results, on average, in an increase of 5.78% of WINS in a teams regular season. Next is STRIKEOUTS. Each increase of 93 STRIKEOUTS, results, on average, in a decrease of 1.83% of WINS in a teams regular season. And finally, WALKS. Each increase of 153 WALKS, results, on average, in an increase of 2.43% of WINS in a team's regular season.">
    
</p>


# Utilizing the Model

## Check SF Giants 2022 Regular Season Prediction
Check the models prediction on SF Giants 2022 regular season

```

The model's prediction for the percentage of wins for the
SF Giant's 2022 regular season is 47.7539325923143%.

The SF Giant's actual win percentage for the 2022 regular
season is 50.0%.

This model's error is ~2.25%

```

<p align="center" width="100%">
    
<img src="/images/ss_examples/22_reg_season_notes.png" alt="blue note box: This model under-predicted the SF Giants 2022 regular season by 2.25% (note this is within the 5.81 RMSE range).">
    
</p>

## Make SF Giants 2023 Regular Season Prediction
Below I consolidate the 2022 regular season stats from players on this year's (2023) roster, aggregate all the hitters into a single team (in the same way I did with the 5 seasons of teams), then use the cumulative team stats to make a prediction for the SF Giants 2023 regular season percentage of wins. 

One obvious issue is that this year's players played on several teams last year, artificially raising the games played. This could have a negative impact on the prediction as these features are strongly correlated with runs and therefore win percentage. I will account for this by taking the mean number of games played over the 5 seasons of data I have collected, divide by the team's number of games played, and multiply all features by the product; this will offset the teams inflated stats. 


### Make prediction for 2023 win percentage using raw stats

```

67.1040865530805

```

<p align="center" width="100%">
    
<img src="/images/ss_examples/yellow_warning_2023_pred.png" alt="yellow warning box: There is an inherent limitation to this prediction in that it is using last year's stats. I cannot get around this limitation. There is a second limitation, the number of games played is inflated due to players playing on so many teams last year. I can attempt to compensate for the resulting inflated stats by taking the mean number of games played from the last 5 seasons and dividing it by the teams total games played in 2022. This will give me a conversion rate to multiple each stat by.">
    
</p>

###  Add regulation to inflated numbers in aggrigated team stats

I use the mean of `Games Played Sum` rather than `At Bats Sum` because at bats could be artificially inflated by longer games. A reminder that `Games Played Sum` is an aggregated team stat, and is the cumulative sum of games played by the players on a team. I suppose it is pertinent to specify, this cumulative mean only includes MLB games and not Minor league appearances as does my SF Giants 2023 roster stats.


### Final prediciton for 2023 SF Giants regular season win percentage


```

60.18449553514759

```

<p align="center" width="100%">
    
<img src="/images/ss_examples/final_2023_reg_season_pred.png" alt="blue note box: This model predicts the SF Giants current roster will deliver wins in 60.18% of their 2023 regular season games. To put this into perspective, this is a 10% improvement over last year's 50% wins. In fact, this is the second highest win percentage in the last 5 seasons, beaten only by the 2021 season, when the SF Giants were 1st NL West with 66.05%.
With this same format, I could feed in fantasy rosters, of any set of players, and make predictions on next year's win percentage.">
    
</p>

## Make SF Giants Fantasy Roster Prediction
The above is only part of utilizing the model. It's real value comes form checking how roster change ups can potentially impact a teams win percentage. Below is a simple example of how the model can be used for recruitment and testing hypothetical rosters.

###  Recruitment
The model showed that the 2 most significant stats that impact a teams win percentage are `Runs` and `Strikeouts`, with `Runs` having a positive impact and `Strikeouts` having a negative impact. This means the most impactful players will have not just high number of runs but will also have, relative to their number of runs, low strikeouts. So rather then searching for just the most runs or the least strikeouts, I want to find the players with the greatest number of runs after strikeouts are subtracted.

Below is a df of an example of 3 recruitment recommendations based on the above mentioned difference between `Runs` and `Strikeouts`. 

<p align="center" width="100%">
    
<img src="/images/ss_examples/RSO_top_3.png" alt="df of an example of 3 recruitment recommendations based on the above mentioned difference between `Runs` and `Strikeouts`.">
    
<img src="/images/ss_examples/blue_recruitment_notes2.png" alt="blue note box: From the web scrapped minor league qualified players, these three players have the highest number of `Runs` after `Strikeouts` are removed. They will likely have the most significant positive impact on a team's win percentage. Note this is a simplified version of selecting players intended only to demonstrate how the model can be used to show the impact of a roster change on a team's win percentage. It may be beneficial to consolidate players' stats from all the minor league teams they have played on, but without extensive knowledge of all the players, it would be dangerous to run a similar code as above that used players names to consolidate stat, as there may be players with the same name. If that knowledge were made available to me I could go back and scrap minor stats WITHOUT limiting the player pool to 'qualified players only' then consolidate a players stats into one annual total (including all minor league teams a player has on). Furthermore, it would be most impactful to select the 3 player stats from the model and multiply each relevant stat by its converted coeff, aggregate the three stats to get the exact impact a player will have on a teams win percentage, then rank from most significant to least.">
    
</p>

### Final win % prediciton for 2023 SF Giants regular season Fantasy Roster 
With the above 3 recruitment recommendations we can swap out 3 of the current roster players - 1 center fielder, 1 3rd baseman, and 1 Catcher. To do this I narrow down the team df to just those positions, then calculate those players `Runs` minus `Strikeouts`. I recalculate a regularization stat to multiply each feature by, then I make the final fantasy roster win percentage. 

```
 
68.34796889112899
 
```

## Compare Current Roster with Fantasy Roster

```
 
The model's regular season win percentage predictions:
 
CURRENT SF Giants roster: 60.18%.
FANTASY SF Giants roster: 68.35%.
 
```

<p align="center" width="100%">
    
<img src="/images/ss_examples/blue_compare_roster_prediction_notes.png" alt="blue note box: By swapping out 3 players the teams predicted win percentage goes up ~8%, from ~60.18% to ~68.35%">
    
</p>


# Perspective of Win Percentages 
To put these win % predictions into perspective, I created a dataframe with world series winners for 2022, 2021, 2019, 2018, and 2017. I include just the three predictive features from the model and teams regular season win percentage.  I also include the SF Giants win % of each respective year for comparison. 

<p align="center" width="100%">
    
<img src="/images/ss_examples/wolrd_series_winners_and_giants.png" alt="df with 5 seasons of world series winners regular season win percentages and the sf giants win percentage for those same 5 seasons">
 
<img src="/images/ss_examples/compare_these_df_notes.png" alt="blue note box: The grey rows are the World Series winners, the white rows are the SF Giants team stats. You can see, with the exception of 2021, the world series winners had regular season win percentages of 10 - 22 percent more than the SF Giants. 2021 the Giants were 1st in National League West, eventually losing the National League Division Series (3-2) to the Dodgers who had a regular season win % of 65.43, less than 1% less than SF Giants.">
    
</p>

# Conclusions on Final Model

## Statsmodel OLS Summary Report

<p align="center" width="100%">
    
<img src="/images/ss_examples/OLS_summary_final.png" alt="statsmodel OLS summary report">
    
</p>


### P Values 

To determine if an observed outcome is statistically significant, we look at the P values; in this case they are all below .05 indicating that there **_is_** a relationship between the associated coefficient and a teams percentage of wins in their regular season games. 

### R squared

This final linear regression model is able to explain 72.7% of the variability observed in regular season percentage of wins.

### Adjusted R squared

The adjusted r squared is essentially the same as the r squared, just 0.7% difference, so we can be confident, as stated above, in the 72.7% reliability of this model.

###  F statistic and Prob(f-statistic)
 
The Prob (F-statistic) of 1.46e-32 tells us, there is 0% chance that any experimentally observed difference is due to chance alone.


### Cond. No

This summary report give a condition number of 3.65, well below 10, indicating there are no multicollinearity issues.


## Limitations
There are 2 limitations in this current model:
- The amount of data
  - While this data goes back 5 seasons there are only 150 data points. These 150 data points were aggregated from nearly 6,000 individual player stats, and over 12,000 games played, however, the final dataset was just 150 data points.
- The statistics used
  - This model only took into account players hitting stats. I did not web scrape any fielding or pitching stats. Intuitively, both impact a teams performance, but they were not part of this model's training data. 


# Final Recommendations

A teams combined RUNS has the most significant impact on their win percentage over a regular season, followed by STRIKEOUTS, then WALKS. 

Below are the direct interpretations from the statsmodel summary report, (because the data was scaled the coeffients are in each feature's standard deviation):
- **For every 117.75 runs, a teams regular season win percentage increases, on average, by ~5.78%**
- **For every 152.97 walks, a teams regular season win percentage increases, on average, by ~2.43%**
- **For every 93.45 strikeout, a teams regular season win percentage _decreases_, on average, by ~1.83%**

Converted to more standardized variables:
- **For every 1 run, a teams regular season win percentage increases, on average, by ~0.0491%**
- **For every 1 walk, a teams regular season win percentage increases, on average, by ~0.0159%**
- **For every 1 strikeout, a teams regular season win percentage _decreases_, on average, by ~0.0195%**

The specificity of the coefficient translation can be helpful when comparing or narrowing down potential recruits. Additionally, with a root mean squared error of just 5.8, this model is, on average, off by only ~5.8% in it's predictions. This shows that the predictive capabilities could prove to be useful for testing/comparing fantasy/hypothetical rosters in order to gain insight in how a team as a whole will perform throughout the regular season.


# Next Steps

- Webscrap more seasons
- Expand to including pitching and feilding stats
- Combine minor and major stats for player annual stats to win percentage
- Use prior year stats to predict post year win percentage
- Multiply each feature by it's converted coeff then aggregated the three features to get the exact impact a player will have on a teams win percentage. 


# Thank You

Let's work together,

- Email: cassigroesbeck@emailplace.com
- GitHub: [@AgathaZareth](https://github.com/AgathaZareth)
- LinkedIn: [Cassarra Groesbeck](https://www.linkedin.com/in/cassarra-groesbeck-a64b75229)

