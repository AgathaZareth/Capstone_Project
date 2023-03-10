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

The data comes from web scraping [MLB.com](https://www.mlb.com/stats/), [MiLB.com](https://www.milb.com), [TheBaseballCube.com](https://thebaseballcube.com), and [D1Baseball.com](https://d1baseball.com). 

Major League Player and Game Stats - [MLB.com](https://www.mlb.com/stats/):
 - 5 regular season player hitting stats 
   -  To avoid collinearity issues down the line I only added statistics that did not have any direct relationship to other stats, i.e. I did not include things that already combined other stats. For example, `batting average` combines `hits` and `at bats` so I included `hits` and `at bats` but left out `batting average`. In summary, any player-stat with a formula was left out. 
 - Data on each game of the 5 regular seasons
   - who played who, where they played, date of game, and each teams score
 
Minor League Triple-A Player Stats - [MiLB.com](https://www.milb.com):
- 2022 player hitting stats. This website defaults to showing qualified players only, so the resulting data frame is much smaller compared to the Major and Collegiate tables. 

Collegiate Division-1 Player Stats - [TheBaseballCube.com](https://thebaseballcube.com), and [D1Baseball.com](https://d1baseball.com):
- 2022 player hitting stats


---
---

# Linear Regression Modeling - Scaled Data

## Build Baseline and Final model

The baseline model takes in the most correlated feature with `win %`. Final model was created by iteratively adding features based on the features p value.

Small print outs of the following are printed for each:

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

# Can Lasso or Ridge regression improve predictive capabilities?

Now that I have my Linear Regression model for interpretability, I can try to boost the predictive capacity by using Lasso and Ridge regression. Both are regularization techniques, one performing L1 regularization, and the other L2.??

Lasso Regression performs L1 regularization. L1 regularization takes the *absolute values* of the weights, so the cost only increases linearly. Lasso??uses shrinkage, i.e. data values shrink towards a central point as the mean.??

Ridge Regression performs L2 regularization. L2 regularization takes the *square* of the weights, so the cost of outliers increases exponentially. Ridge regression is good to use if there is multicollinearity between the??features of your data. When issues of multicollinearity occur, least-squares are unbiased, and variances are large, this results in predicted values being far away from the actual values.??


## Get RMSE's for *all* features
Use entire `X_train` (all features) to check different RMSE's with 3 different models.

```

RidgeCV() rmse:			6.073710918791899
LassoCV() rmse:			6.0315888246948015
LinearRegression() rmse:	6.600711237023368

```


## Plot optimal alphas (lambdas)

This is more of a brute force approach, rather than trusting the automatic alpha selection, I want to check a range of alphas and plot the MSE for each. Then I will compare the RMSE metrics to see if there is any improvement. If not I will use the auto-selected alpha value. 

*Original code for plots can be found [HERE](https://github.com/learn-co-curriculum/dsc-ridge-and-lasso-regression-lab/tree/solution#finding-an-optimal-alpha) from "Ridge and Lasso Regression - Lab" solution branch*


### Lasso

<p align="center" width="100%">
    
<img src="/images/ss_examples/lasso_plot_alpha.png" alt="plot of alpha values and mean squared errors for train and test data using lasso regression. Optimal alpha determined by the point at which the train and test MSE's are closest.">
    
</p>


### Ridge

<p align="center" width="100%">
    
<img src="/images/ss_examples/ridge_plot_alpha.png" alt="plot of alpha values and mean squared errors for train and test data using ridge regression. Optimal alpha determined by the point at which the train and test MSE's are closest.">
    
</p>

### Test optimal alphas from plots

Lasso alpha = 0
```
6.545520231720073
```

Ridge alpha = 8
```
6.074428900049834
```

<p align="center" width="100%">
    
<img src="/images/ss_examples/plot_alphas_notes.png" alt="blue note box: There is no improvement using a manual search. I will stick to the automatically selected values.">
    
</p>



## Get additional RMSE's comparisons

Reduce `X_train` to just the cumulative features to see if there is an improvement.
- `Hits Sum`, 
- `Runs Batted In Sum`
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

```

RidgeCV() rmse:			6.048666345049738
LassoCV() rmse:			5.84469104287651
LinearRegression() rmse:	6.000390901490906

```

Reduce `X_train` to just the cumulative features left AFTER heatmap cuts.
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

```

RidgeCV() rmse:			6.0598572600893785
LassoCV() rmse:			5.866461018913089
LinearRegression() rmse:	6.076677429513871

```

Reduce `X_train` to just the features that made it into the final Linear Regression model.
- `Runs Sum`
- `Walks Sum`
- `Strikeouts Sum`

```

RidgeCV() rmse:			5.816366647123051
LassoCV() rmse:			5.814338819118407
LinearRegression() rmse:	5.812752574210284

```

Reduce `X_train` to just the one most correlated feature to `% wins` 
- `Runs Sum`

```

RidgeCV() rmse:			6.373948545890253
LassoCV() rmse:			6.3756014274341855
LinearRegression() rmse:	6.375864829436179

```


<p align="center" width="100%">
    
<img src="/images/ss_examples/get_additional_rmses_notes2.png" alt="blue note box: There is too much noise with other features added to improve the error of the model. All 3 models performed best with just the 3 features used in Linear Regression final model, with Lasso Regression using all 12 cumulative features coming in a very close 4th place. Ultimatly, Linear Regression model with the 3 feats had lowest error.
CONCLUSION: Linear Regression model with 3 features - Runs Sum, Walks Sum, Strikeouts Sum - is the best model for both predictive and inferential purposes.">
    
</p>


# Linear Regression Final Model Assumption checks

Regression is a powerful analysis however, if some of the necessary assumptions are not satisfied, regression makes biased and unreliable predictions. Below I check the following 4 assumptions:
- Independence Assumption
- Linearity Assumption
- Homoscedasticity Assumption
- Normality Assumption

I give brief explanations of these assumption checks at each header but you can also visit:
- [HERE](https://github.com/learn-co-curriculum/dsc-regression-assumptions#about-regression-assumptions) for a short summary of Linearity, Homoscedasticity, and Normality Assumptions. 

## Independence Assumption

Because I am using this model for both inferential and predictive purposes I need to ensure that each observation is independent of the others. A violation of the independence assumption results in incorrect confidence intervals and p-values, it can essentially be thought of as a kind of double-counting in the model and it can produce estimates of the regression coefficients that are not statistically significant. 

This article ["How to detect and deal with Multicollinearity"](https://towardsdatascience.com/how-to-detect-and-deal-with-multicollinearity-9e02b18695f1) does a really good job of explaining the differences between using??correlations vs VIF, 

>A correlation plot can be used to identify the correlation or bivariate relationship between two independent variables whereas VIF is used to identify the correlation of one independent variable with a group of other variables. Hence, it is preferred to use VIF for better understanding.
>
>- VIF = 1 ??? No correlation
>- VIF = 1 to 5 ??? Moderate correlation
>- VIF >10 ??? High correlation

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


## Linearity Assumption

The linearity assumption requires that there is a linear relationship between the response variable (Y) and predictor (X). Linear means that the change in Y by 1-unit change in X, is constant. If you were to try to fit a linear model to a non-linear data set, OLS would fail to capture the trend mathematically, resulting in an inaccurate relationship. This will also result in erroneous predictions on an unseen data set.

<p align="center" width="100%">

<img src="/images/ss_examples/regplots_3_feats.png" alt="regplots of three features that made it into final model.">
    
<img src="/images/ss_examples/linearity_assumption_notes.png" alt="green success box: The three features show varying degrees of a linear relationship with the target feature. The assuption of linearity is satisfied.">
    
</p>



## Homoscedasticity Assumption

Homoscedasticity indicates that a dependent variable's variability is equal across values of the independent variable. A scatter plot is a good way to check whether the data are homoscedastic (meaning the residuals are equal across the regression line). 

<p align="center" width="100%">

<img src="/images/ss_examples/residuals_vs_predicted.png" alt="scatter plot of residuals vs predicted y values.">
    
<img src="/images/ss_examples/homoscedasticity_assumption_notes.png" alt="green success box: The scatter plot of residuals do not show any kind of pattern and are equally distributed. The assumption of homoscedasticity is satisfied.">
    
</p>

## Normality Assumption

The normality assumption states that the model residuals should follow a normal distribution. This can be viewed with either a histogram or a QQ Plot of the residuals. I prefer to use both to fully understand the distribution. 


<p align="center" width="100%">

<img src="/images/ss_examples/histo_qq_resids.png" alt="histogram and qq plot of residuals.">
    
<img src="/images/ss_examples/normality_assumption_notes.png" alt="green success box: The histogram and qq plots show the resdiuals have a roughly normal distribution. The assumption of normality is satisfied.">
    
</p>

# Interpreting Linear Regression Model Results

<p align="center" width="100%">

<img src="/images/ss_examples/coeffs_to_df.png" alt="screen shot of coeffs_to_df, gives coefficients of independent variables from the model and how they increase or decrease percent of wins in a regular season.">
 
<img src="/images/ss_examples/coeff_to_df_notes2.png" alt="blue note box: This df shows, RUNS is the most significant feature when predicting a teams win percentage. Each increase of 118 RUNS, results, on average, in an increase of 5.78% of WINS in a teams regular season. Next is STRIKEOUTS. Each increase of 93 STRIKEOUTS, results, on average, in a decrease of 1.83% of WINS in a teams regular season. And finally, WALKS. Each increase of 153 WALKS, results, on average, in an increase of 2.43% of WINS in a team's regular season.">
    
</p>


# Utilizing the model

## Check SF Giants 2022 regular season prediction
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

## Make SF Giants 2023 regular season prediction
Below I consolidate the 2022 regular season stats from players on this year's (2023) roster. I then aggregate??all the hitters into a single team in the same way I did with the 5 seasons of teams. I use this to make a prediction for the SF Giants 2023 regular season percentage of wins.??

One obvious issue is that this year's players played on several teams last year, artificially raising the games played and at bats. This could have a negative impact on the prediction as these features are strongly correlated with runs and therefore win percentage.??I will account for this by taking the mean number of games played, over the 5 seasons of data I have collected, to offset the teams inflated numbers. 


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

## Make SF Giants fantasy roster prediction
The above is only part of utilizing the model. It's real value comes form checking how roster change ups can potentially impact a teams win percentage. Below is a simple example of how the model can be used for recruitment and testing hypothetical rosters.

###  Recruitment
The model showed that the 2 most significant stats that impact a teams win percentage are `Runs` and `Strikeouts`, with `Runs` having a positive impact and `Strikeouts` having a negative impact. This means the most impactful players will have not just high number of runs but will also have, relative to their number of runs, low strikeouts. So rather then searching for just the most runs or the least strikeouts, I want to find the players with the greatest number of runs after strikeouts are subtracted.

Below is a df of an example of 3 recruitment recommendations based on the above mentioned difference between `Runs` and `Strikeouts`. 

<p align="center" width="100%">
    
<img src="/images/ss_examples/RSO_top_3.png" alt="df of an example of 3 recruitment recommendations based on the above mentioned difference between `Runs` and `Strikeouts`">
    
<img src="/images/ss_examples/blue_recruitment_notes.png" alt="blue note box: From the web scrapped minor league qualified players, these three players have the highest number of `Runs` after `Strikeouts` are removed. They will likely have the most significant positive impact on a team's win percentage. Note this is a simplified version of selecting players intended only to demonstrate how the model can be used to show the impact of a roster change on a team's win percentage.
   
It may be beneficial to consolidate players' stats from all the minor league teams they have played on, but without extensive knowledge of all the players, it would be dangerous to run a similar code as above that used players names to consolidate stat, as there may be players with the same name. If that knowledge were made available to me I could go back and scrap minor stats WITHOUT limiting the player pool to 'qualified players only' then consolidate a players stats into one annual total (including all minor league teams a player has on).  
   
Furthermore, it would be most impactful to select the 3 player stats from the model and multiply each relevant stat by its converted coeff, aggregate the three stats to get the exact impact a player will have on a teams win percentage, then rank from most significant to least.">
    
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


# Perspective of win percentages 
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


### P Values and confidence intervals

Up until this point p-values have been the primary focus, with all features included or excluded in or from the model based on this value, more specifically, in this case, the threshold of 0.05 or greater. In the summary report, p-values are in the column `P > |t|` and rounded to 3 digits. 

Applied to a regression model, p-values associated with coefficient estimates indicate the probability of observing the associated coefficient given that the null-hypothesis is true. In this case the null hypothesis is:
There is NO relationship between the associated coefficient - "a teams cumulative runs", "a teams cumulative walks", and "a teams cumulative strikeouts" - and the the teams percentage of wins in their regular season games. 
This null hypothesis can be rejected when the p value is below 0.05.  

Rejecting the null hypothesis at an alpha level of 0.05 is the equivalent for having a 95% confidence interval around the coefficient that does not include zero. The confidence intervals are in the two last columns to the right of `P > |t|`, in this case it is 95%, derived from the shown range `[0.025 0.975]`. Using `Runs Sum` as an example: The confidence interval for `Runs Sum` is [4.373, 7.194] meaning that there is a 95% chance that the actual coefficient value is in that range. Note this is why it is important this range does not span 0, this would indicate an uncertainty of having either a positive or negative correlation, it can't be both.

As the measurement of how likely a coefficient is measured by our model through chance, p-values are undoubtedly important, however, they are not the only important metric when analyzing the statsmodel summary report.

### R squared

R-squared is the measurement of how much of the independent variable is explained by changes in our dependent variables. In this case, the r squared value of 0.727 tells us: 

**This final linear regression model is able to explain 72.7% of the variability observed in regular season percentage of wins.** 
        
Because r squared will always increase as features are added (in this case 3 as shown in 'Df Model:' on the summary report) we should also look at the adjusted r squared to get a better understanding of how the model is performing.

### Adjusted R squared

Adjusted r squared takes into account the number of features in the model by penalizing the R-squared formula based on the number of variables. If the two were significantly different it could be that some variables are not contributing to your model???s R-squared properly. In this case:

**The adjusted r squared is essentially the same as the r squared, just 0.7% difference, so we can be confident, as stated above in `10A - b. R squared`, in the 72% reliability of this model.**

###  F statistic and Prob(f-statistic)

The f statistic is also important. More easily understood is the prob(f-statistic), it uses the F statistic to tell the accuracy of the null hypothesis, or whether it is accurate that the variables??? effects are 0. In this case, it is telling us there is 0.00% of this. 

**The Prob (F-statistic) of 1.46e-32 tells us, there is 0% chance that any experimentally observed difference is due to chance alone.**

This essentially translates to: an underlying causal relationship __*does*__ exist between the 3 independent variables used in the model and the dependent variable of % of wins.


### Cond. No

A condition number of 10-30 indicates multicollinearity, and a condition number above 30 indicates strong multicollinearity. 

**This summary report give a condition number of 3.65, well below 10, indicating there are no multicollinearity issues.**

This is important because multicollinearity among independent variables will result in less reliable statistical inferences and making it hard to determine how the independent variables influence the dependent variable individually. 

## Limitations
There are 2 limitations in this current model:
- The amount of data
  - While this data goes back 5 seasons there are only 150 data points. These 150 data points were aggregated from over 5,000 individual player stats, but the final dataset was just 150 data points.
- The statistics used
  - This model only took into account players hitting stats. I did not web scrape any fielding or pitching stats. Intuitively, both impact a teams performance, but they were not part of this model's training data. 


# Final Recommendations

A teams combined RUNS has the most significant impact on their win percentage over a regular season, followed by STRIKEOUTS, then WALKS. 

Below are the direct interpretations from the statsmodel summary report, (because the data was scaled the coeffients are in each feature's standard deviation):
- For every 117.75 runs, a teams regular season win percentage increases, on average, by ~5.78%
- For every 152.97 walks, a teams regular season win percentage increases, on average, by ~2.43%
- For every 93.45 strikeout, a teams regular season win percentage _**decreases**_, on average, by ~1.83%

Converted to more standardized variables:
- For every 1 run, a teams regular season win percentage increases, on average, by ~0.0491%
- For every 1 walk, a teams regular season win percentage increases, on average, by ~0.0159%
- For every 1 strikeout, a teams regular season win percentage _**decreases**_, on average, by ~0.0195%

These findings do not shed any revolutionary??knowledge, however, the specificity of the coefficient translation can be helpful when comparing or narrowing down potential recruits. Furthemore, with a root mean squared error of just 5.8, this model is, on average, off by only ~5.8% in it's predictions, meaning the predictive capabilities??could prove to be useful for testing/comparing fantasy/hypothetical rosters to get some insight in how a team as a whole will perform throughout the regular season.


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












---
---


# Summary of Exploratory Data Analysis of Raw Data

An important consideration when using multiple predictors in any machine learning model is the scale of these features. `pandas.DataFrame.describe` shows a huge range in each of the independent features. Variables of vastly different scales can impact the influence over the model, therefore, it is best practice to normalize the scale of all features before feeding the data into a machine learning algorithm. This will ensure the features with larger numeric values are not unfairly weighted by the model. When deciding which method to use when scaling, it can be helpful to understand the distribution of values. 

<p align="center" width="100%">
<img src="/images/ss_examples/raw_data_distribution_of_values.png" alt="histogram thumbnail tiles of each feature">

<img src="/images/ss_examples/blue_note_box_raw_histo_tiles.png" alt="blue note box: Most features have a roughly normal distribution. I don't see any extreme skewness that might justify logging a variable.">

</p>



# Summary of Preprocessing

To avoid any potential data leakage, I did a train test split before altering the data in any way.`sklearn.model_selection.train_test_split` (documentation [HERE](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html))

## Scale Data

`sklearn.preprocessing.StandardScaler` default standardization (documentation [HERE](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)). Feature scaling is a method used to normalize the range of the independent variables of data. A machine learning algorithm can only see numbers; this means, if there is a vast difference in the feature ranges (as there is with this data, demonstrated in 'Summary of Exploratory Data Analysis of Raw Data') it makes the underlying assumption that higher ranging numbers have superiority of some sort and these more significant numbers start playing a more decisive role while training the model. 

Choosing which method to use depends on the distribution of your data.  A couple of relevant generalizations are: 

- Standardization may be used when data represent Gaussian Distribution, while Normalization is great with Non-Gaussian Distribution
- Impact of Outliers is very high in Normalization


### Explore the Scaling Effect on Training Data


#### Boxplots
Boxplots (documentation [HERE](https://seaborn.pydata.org/generated/seaborn.boxplot.html)) are a great way to see the change in the range of each independent feature before and after standard scaling. 


<p align="center" width="100%">
<img src="/images/ss_examples/boxplot_before_scaling.png" alt="boxplot of data BEFORE scaling">

<img src="/images/ss_examples/boxplot_after_scaling.png" alt="boxplot of data AFTER scaling">

</p>

#### Histograms
An additional way to view the scaling effect is through histograms (documentation [HERE](https://seaborn.pydata.org/generated/seaborn.histplot.html)). If you think of boxplots as a top view of distributions, then you can think of histograms as a side view. Imagine yourself standing on the right hand side of the above boxplots looking down the 0 line.??

<p align="center" width="100%">
<img src="/images/ss_examples/histos_before_scaling.png" alt="histograms of data BEFORE scaling">

<img src="/images/ss_examples/histos_after_scaling.png" alt="histograms of data AFTER scaling">

<img src="/images/ss_examples/vizualization_notes.png" alt="blue note box: You can see the means of all the independent variables are roughly around 0 and they all have roughly the same min and max values. From the boxplots, you can also see there are a few outliers, primarily in the team averaged variables.">
    
</p>


# Summary of Feature Reduction
## Regplots

Now that data is scaled I can use regplots (documentation [HERE](https://seaborn.pydata.org/generated/seaborn.regplot.html)) to view if there is a linear relationship between the different statistics and a teams win percentage, and to show which features have a 'cleaner' linear relationship with target variable.  

<p align="center" width="100%">
<img src="/images/ss_examples/regplot_all_feats.png" alt="regplots of all 24 independent variables">

<img src="/images/ss_examples/regplot_notes.png" alt="blue note box: It appears to me that the cumulative sum totals have either the same or even cleaner linear relationships with the target variable. You can see this with just the scatter plots alone but the red shaded band is perhaps a better representation. This red line shows what a linear model might look like, with the shaded part being the confidence interval. The more 'noise' the wider that red shaded area. A perfectly confident model would be a clean red line without a shaded band. I will use heatmaps to investigate further.">
    
</p>

## Heatmaps

The regplots show the relationship between the features and target variable. The heatmap (documentation [HERE](https://seaborn.pydata.org/generated/seaborn.heatmap.html)) will show the correlations between all of the numeric values (in this case, all the features) in our data. The x and y axis labels indicate the pair of values that are being compared, and the color and the number are both representing the correlation. Color is used here to make it easier to find the largest/smallest numbers. The bottom row is particularly important because it shows all the features correlation with the target variable but I am also looking for strong correlation between the independent variables.  


### Heatmap of Cumulative Features

<p align="center" width="100%">
<img src="/images/ss_examples/heatmap_cumulative.png" alt="heatmap of the 12 cumulative independent variables and the target variable">
    
</p>

### Heatmap of Averaged Features

<p align="center" width="100%">
<img src="/images/ss_examples/heatmap_averaged.png" alt="heatmap of the 12 averaged independent variables and the target variable">
    
<img src="/images/ss_examples/heatmap_notes.png" alt="blue note box: Interestingly, the averaged variables have a few features with a stronger correlation with the target variable than those of the corresponding cumulative variables. Overall however, the cumulative features seem to have stronger correlations. In addition, there are a lot more .9 (or above) inter-feature correlations among the averaged variables. The averaged variables have 18 inter correlated pairs of .9 or greater, while the cumulative variables have only 5.">
    
</p>

## Selecting Features to Drop

Based on the Heatmap and Regplots I will *keep* the cumulative sum features. I need to eliminate the highly correlated features while keeping as much data as possible. This means I need to remove the least amount of features possible. `Runs Sum` has the highest correlation with `% wins` so I want to keep this feature. 


<p align="center" width="100%">
<img src="/images/elimination_tables.png" alt="table showing elimination process">
    
</p>

By eliminating `Hits Sum` & `Runs Batted In Sum` I remove all the pairs with a correlation of .9 or greater. This leaves:
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


## Pairplot
The final visualization to check inter feature correlation is a pairplot (documentation [HERE](https://seaborn.pydata.org/generated/seaborn.pairplot.html)). Pairplots are great ways to see the relationship between two features using scatter plots. Pariplots also show the distribution of each feature across the diagonal. These histograms have already been plotted but here we can see just the features I have decided to keep.

<p align="center" width="100%">
<img src="/images/ss_examples/pairplot.png" alt="pairplot of remaining independent variables and target feature">
    
<img src="/images/ss_examples/pairplot_notes.png" alt="blue note box: I can see there are still quite a bit of features that are correlated with eachother. These will likely be filtered out by p value when modeling.">
    
</p>




# Linear Regression Modeling - Scaled Data

## Build Baseline and Final model

This is all wrapped up in a beautiful function that creates a baseline model by determining the highest positively correlated feature with `% wins`. Then it iteratively adds features based on the features p value. See `build_models` docstring for more info. 

Determines highest correlated feature, creates and prints a baseline
    model using `model_it_small` function. 
    
    From there it determines the features not used in baseline model (or 
    current model), and checks p values of each feature, individually added 
    to base model. From said features, it determines the lowest p value.
    If that lowest p value is under 0.05 the associated feature is added to 
    the current model df and the process is repeated until there are no features
    that can be added, ie all p values are above 0.05, or all features have been 
    used.
    
    It then prints/returns `model_it_small` of all features with pvalues
    less than 0.05. 
    
    Output
    ----------
    Histogram of distribution of residuals
    
    QQ plot
    
    Prints r-squared and adjusted r-squared from statsmodel,
        and kfold crossvalidation with 5 splits
    
    
    Returns
    ----------
    Second half of .summary() table (coefficients, std error, t, p, and 
        ci values)


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

This article ["How to detect and deal with Multicollinearity"](https://towardsdatascience.com/how-to-detect-and-deal-with-multicollinearity-9e02b18695f1) does a really good job of explaining the differences between using??correlations vs VIF, 

>A correlation plot can be used to identify the correlation or bivariate relationship between two independent variables whereas VIF is used to identify the correlation of one independent variable with a group of other variables. Hence, it is preferred to use VIF for better understanding.
>
>- VIF = 1 ??? No correlation
>- VIF = 1 to 5 ??? Moderate correlation
>- VIF >10 ??? High correlation

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
    
<img src="/images/ss_examples/linearity_assumption_notes.png" alt="green success box: The three features show varying degrees of a linear relationship with the target feature. The assuption of linearity is satisfied.">
    
</p>



#### 8Bb - iii. Homoscedasticity Assumption

Homoscedasticity indicates that a dependent variable's variability is equal across values of the independent variable. A scatter plot is a good way to check whether the data are homoscedastic (meaning the residuals are equal across the regression line). 

<p align="center" width="100%">

<img src="/images/ss_examples/residuals_vs_predicted.png" alt="scatter plot of residuals vs predicted y values.">
    
<img src="/images/ss_examples/homoscedasticity_assumption_notes.png" alt="green success box: The scatter plot of residuals do not show any kind of pattern and are equally distributed. The assumption of homoscedasticity is satisfied.">
    
</p>

#### 8Bb - iv. Normality Assumption

The normality assumption states that the model residuals should follow a normal distribution. This can be viewed with either a histogram or a QQ Plot of the residuals. I prefer to use both to fully understand the distribution. 


<p align="center" width="100%">

<img src="/images/ss_examples/histo_qq_resids.png" alt="histogram and qq plot of residuals.">
    
<img src="/images/ss_examples/normality_assumption_notes.png" alt="green success box: The histogram and qq plots show the resdiuals have a roughly normal distribution. The assumption of normality is satisfied.">
    
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
 
<img src="/images/ss_examples/coeff_to_df_notes.png" alt="blue note box: This df shows, RUNS is the most significant feature when predicting a teams win percentage. Each increase of 118 RUNS, results, on average, in an increase of 5.78% of WINS in a teams regular season. Next is WALKS. Each increase of 153 WALKS, results, on average, in an increase of 2.43% of WINS in a team's regular season. And finally, STRIKEOUTS. Each increase of 93 STRIKEOUTS, results, on average, in a decrease of 1.83% of WINS in a teams regular season..">
    
</p>


To put this into perspective, I will create a dataframe with world series winners for 2022, 2021, 2019, 2018, and 2017. I will include just the three predictive features from the model and the teams regular season win percentage.  I will also include the SF Giants for each respective year for comparison. 

<p align="center" width="100%">
    
<img src="/images/ss_examples/wolrd_series_winners_and_giants.png" alt="df with 5 seasons of world series winners regular season win percentages and the sf giants win percentage for those same 5 seasons">
 
<img src="/images/ss_examples/compare_these_df_notes.png" alt="blue note box: The grey rows are the World Series winners, the white rows are the SF Giants team stats. You can see, with the exception of 2021, the world series winners had regular season win percentages of 10 - 22 percent more than the SF Giants. 2021 the Giants were 1st in National League West, eventually losing the National League Division Series (3-2) to the Dodgers who had a regular season win % of 65.43, less than 1% less than SF Giants.">
    
</p>


## 8 - D. Using the model for predictions

### 8D - a. Check SF Giants 2022 regular season prediction
Check the models prediction on SF Giants 22 regular season

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

### 8D - b. Make SF Giants 2023 regular season prediction

Below I will consolidate the 2022 regular season stats from players on this year's (2023) roster. I will then aggregate??all the hitters into a single team in the same way I did with the 5 seasons of teams. I will use this to make a prediction for the SF Giants 2023 regular season percentage of wins.??

One obvious issue is that this year's players played on several teams last year, artificially raising the games played and at bats. This could have a negative impact on the prediction as these features are strongly correlated with runs and therefore win percentage.??I will account for this by taking the mean number of games played, over the 5 seasons of data I have collected, to offset the teams inflated numbers. 


#### 8Db - i. Get 2023 roster (web scrap)

Below I will do one final web scraping, hopefully, to get the 2023 SF Giants roster.

  - **To bypass this step when re running notebook skip to next cell**
  
I am scraping from [Baseball America](https://www.baseballamerica.com/teams/1018/san-francisco-giants/roster/)

#### 8Db - ii. Load 2023 roster (pickled roster df)

Load the above pickled df.

#### 8Db - iii. Trim to get just the hitters

Drop the pitchers, and header rows, to get just the hitters.

#### 8Db - iv. Load 2022 player stats

Load 2022 stats from Major, Minor, and Collegiate leagues. 

#### 8Db - v. Find players on 2023 SF roster

Loop through 2022 stats and find players on SF Giants 2023 roster.

#### 8Db - vi. [Create SF Giants 2023 roster stats df](#Table_of_Contents)

Get SF Giants 2023 hitters (ie not pitchers) stats from 2022

<p align="center" width="100%">
    
<img src="/images/ss_examples/create_sf_giants_2023_roster_stats_df_notes.png" alt="blue note box: Only 'Heliot Ramos' came up in minors (see: 8Db - v. Find players on 2023 SF roster). This means the remaining players were not included in my web scraping of qualified players only. I will need to manually collect those stats and join them to the existing MLB players that I already have. Additionally, several of our 2023 players played on multiple teams last year. If the additional team was a minor league team, those minor league stats do not carry over to their MLB stats. I will have to manually check each player's 2022 stats to ensure I have all relevant data.">
    
</p>


Add 2022 minor league stats for all SF 2023 roster.

Consolidate players stats from all 2022 teams, so that each player only appears once in the 2023 roster, with all their stats from the 2022 season combined. 

#### 8Db - vii. Aggregate??team stats (into 1 line df)

#### 8Db - viii. Make predictions for 2023 win percentage

```

67.1040865530805

```


<p align="center" width="100%">
    
<img src="/images/ss_examples/yellow_warning_2023_pred.png" alt="yellow warning box: There is an inherent limitation to this prediction in that it is using last year's stats. I cannot get around this limitation. There is a second limitation, the number of games played is inflated due to players playing on so many teams last year. I can attempt to compensate for the resulting inflated stats by taking the mean number of games played from the last 5 seasons and dividing it by the teams total games played in 2022. This will give me a conversion rate to multiple each stat by.">
    
</p>

#### 8Db - ix. Add a regulation to inflated numbers in aggrigated team stats

I am using the mean of `Games Played Sum` rather than `At Bats Sum` because at bats could be artificially inflated by longer games. A reminder that `Games Played Sum` is an aggregated team stat, and is the cumulative sum of games played by the players on a team. I suppose it is pertinent to specify, this cumulative mean only includes MLB games and not Minor league appearances as does my SF Giants 2023 roster stats.


#### 8Db - x. Final prediciton for 2023 SF Giants regular season win percentage


```

60.18449553514759

```

<p align="center" width="100%">
    
<img src="/images/ss_examples/final_2023_reg_season_pred.png" alt="blue note box: This model predicts the SF Giants current roster will deliver wins in 60.18% of their 2023 regular season games. To put this into perspective, this is a 10% improvement over last year's 50% wins. In fact, this is the second highest win percentage in the last 5 seasons, beaten only by the 2021 season, when the SF Giants were 1st NL West with 66.05%.
With this same format, I could feed in fantasy rosters, of any set of players, and make predictions on next year's win percentage.">
    
</p>


# 9. Can Lasso or Ridge regression improve predictive capabilities?

Now that I have my Linear Regression model for interpretability, I can try to boost the predictive capacity by using Lasso and Ridge regression. Both are regularization techniques, one performing L1 regularization, and the other L2.??

Lasso Regression performs L1 regularization. L1 regularization takes the *absolute values* of the weights, so the cost only increases linearly. Lasso??uses shrinkage, i.e. data values shrink towards a central point as the mean.??

Ridge Regression performs L2 regularization. L2 regularization takes the *square* of the weights, so the cost of outliers increases exponentially. Ridge regression is good to use if there is multicollinearity between the??features of your data. When issues of multicollinearity occur, least-squares are unbiased, and variances are large, this results in predicted values being far away from the actual values.??


## 9 - A. Re load `MODELING_DF`

To ensure all variables are correct I will start from scratch and reload the `MODELING_DF`

## 9 - B. Re split data

## 9 - C. Create functions

To streamline comparisons between Ridge, Lasso, and LinearRegression, I will make a couple quick little functions. 


### 9C - a. `get_rmse`

This is mostly a helper function. It uses [sklearn.pipeline.Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html), [sklearn.preprocessing.StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler), and [sklearn.metrics.mean_squared_error](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html) to fit a regressor to training data and get RMSE for test data.

### 9C - b. `compare_metrics`

This function uses `get_rmse` to print out rmse for [sklearn.linear_model.LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression), [sklearn.linear_model.LassoCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html), and [sklearn.linear_model.RidgeCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html). 

## 9 - D. Get RMSE's for *all* features
Use `compare_metrics` function with the entire `X_train` (all features) to check different RMSE's with 3 different models.

```

RidgeCV() rmse:			6.073710918791899
LassoCV() rmse:			6.0315888246948015
LinearRegression() rmse:	6.600711237023368

```


## 9 - E. [Plot optimal alphas (lambdas)](#Table_of_Contents)

This is more of a brute force approach, rather than trusting the automatic alpha selection, I want to check a range of alphas and plot the MSE for each. Then I will compare the RMSE metrics to see if there is any improvement. If not I will use the auto-selected alpha value. 

*Original code for plots can be found [HERE](https://github.com/learn-co-curriculum/dsc-ridge-and-lasso-regression-lab/tree/solution#finding-an-optimal-alpha) from "Ridge and Lasso Regression - Lab" solution branch*


### 9E - a. Lasso

<p align="center" width="100%">
    
<img src="/images/ss_examples/lasso_plot_alpha.png" alt="plot of alpha values and mean squared errors for train and test data using lasso regression. Optimal alpha determined by the point at which the train and test MSE's are closest.">
    
</p>


### 9E - b. Ridge

<p align="center" width="100%">
    
<img src="/images/ss_examples/ridge_plot_alpha.png" alt="plot of alpha values and mean squared errors for train and test data using ridge regression. Optimal alpha determined by the point at which the train and test MSE's are closest.">
    
</p>

### 9E - c. Test optimal alphas from plots

Lasso alpha = 0
```
6.545520231720073
```

Ridge alpha = 8
```
6.074428900049834
```

<p align="center" width="100%">
    
<img src="/images/ss_examples/plot_alphas_notes.png" alt="blue note box: There is no improvement using a manual search. I will stick to the automatically selected values.">
    
</p>



## 9 - F. Get additional RMSE's comparisons

Reduce `X_train` to just the cumulative features to see if there is an improvement.
- `Hits Sum`, 
- `Runs Batted In Sum`
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

```

RidgeCV() rmse:			6.048666345049738
LassoCV() rmse:			5.84469104287651
LinearRegression() rmse:	6.000390901490906

```

Reduce `X_train` to just the cumulative features left AFTER heatmap cuts.
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

```

RidgeCV() rmse:			6.0598572600893785
LassoCV() rmse:			5.866461018913089
LinearRegression() rmse:	6.076677429513871

```

Reduce `X_train` to just the features that made it into the final Linear Regression model.
- `Runs Sum`
- `Walks Sum`
- `Strikeouts Sum`

```

RidgeCV() rmse:			5.816366647123051
LassoCV() rmse:			5.814338819118407
LinearRegression() rmse:	5.812752574210284

```

Reduce `X_train` to just the one most correlated feature to `% wins` 
- `Runs Sum`

```

RidgeCV() rmse:			6.373948545890253
LassoCV() rmse:			6.3756014274341855
LinearRegression() rmse:	6.375864829436179

```


<p align="center" width="100%">
    
<img src="/images/ss_examples/get_additional_rmses_notes.png" alt="blue note box: There is too much noise with other features added to improve the error of the model. Lasso reegression did a really good job using all cumulative features, almost the same RMSE as the simpler Linear Regression model.
CONCLUSION: Linear Regression model with 3 features - Runs Sum, Walks Sum, Strikeouts Sum - is the best model for both predictive and inferential purposes.">
    
</p>

# 10. Conclusions on Final Model

## 10 - A. statsmodel OLS summary report

<p align="center" width="100%">
    
<img src="/images/ss_examples/OLS_summary_final.png" alt="statsmodel OLS summary report">
    
</p>

### 10A - a. P Values and confidence intervals

Up until this point p-values have been the primary focus, with all features included or excluded in or from the model based on this value, more specifically, in this case, the threshold of 0.05 or greater. In the summary report, p-values are in the column `P > |t|` and rounded to 3 digits. 

Applied to a regression model, p-values associated with coefficient estimates indicate the probability of observing the associated coefficient given that the null-hypothesis is true. In this case the null hypothesis is:
There is NO relationship between the associated coefficient - "a teams cumulative runs", "a teams cumulative walks", and "a teams cumulative strikeouts" - and the the teams percentage of wins in their regular season games. 
This null hypothesis can be rejected when the p value is below 0.05.  

Rejecting the null hypothesis at an alpha level of 0.05 is the equivalent for having a 95% confidence interval around the coefficient that does not include zero. The confidence intervals are in the two last columns to the right of `P > |t|`, in this case it is 95%, derived from the shown range `[0.025 0.975]`. Using `Runs Sum` as an example: The confidence interval for `Runs Sum` is [4.373, 7.194] meaning that there is a 95% chance that the actual coefficient value is in that range. Note this is why it is important this range does not span 0, this would indicate an uncertainty of having either a positive or negative correlation, it can't be both.

As the measurement of how likely a coefficient is measured by our model through chance, p-values are undoubtedly important, however, they are not the only important metric when analyzing the statsmodel summary report.

### 10A - b. R squared

R-squared is the measurement of how much of the independent variable is explained by changes in our dependent variables. In this case, the r squared value of 0.727 tells us: 

**This final linear regression model is able to explain 72.7% of the variability observed in regular season percentage of wins.** 
        
Because r squared will always increase as features are added (in this case 3 as shown in 'Df Model:' on the summary report) we should also look at the adjusted r squared to get a better understanding of how the model is performing.

### 10A - c. Adjusted R squared

Adjusted r squared takes into account the number of features in the model by penalizing the R-squared formula based on the number of variables. If the two were significantly different it could be that some variables are not contributing to your model???s R-squared properly. In this case:

**The adjusted r squared is essentially the same as the r squared, just 0.7% difference, so we can be confident, as stated above in `10A - b. R squared`, in the 72% reliability of this model.**

### 10A - d. F statistic and Prob(f-statistic)

The f statistic is also important. More easily understood is the prob(f-statistic), it uses the F statistic to tell the accuracy of the null hypothesis, or whether it is accurate that the variables??? effects are 0. In this case, it is telling us there is 0.00% of this. 

**The Prob (F-statistic) of 1.46e-32 tells us, there is 0% chance that any experimentally observed difference is due to chance alone.**

This essentially translates to: an underlying causal relationship __*does*__ exist between the 3 independent variables used in the model and the dependent variable of % of wins.


### 10A - e. Cond. No

A condition number of 10-30 indicates multicollinearity, and a condition number above 30 indicates strong multicollinearity. 

**This summary report give a condition number of 3.65, well below 10, indicating there are no multicollinearity issues.**

This is important because multicollinearity among independent variables will result in less reliable statistical inferences and making it hard to determine how the independent variables influence the dependent variable individually. 

## 10 - B. Limitations
There are 2 limitations in this current model:
- The amount of data
  - While this data goes back 5 seasons there are only 150 data points. These 150 data points were aggregated from over 5,000 individual player stats, but the final dataset was just 150 data points.
- The statistics used
  - This model only took into account players hitting stats. I did not web scrape any fielding or pitching stats. Intuitively, both impact a teams performance, but they were not part of this model's training data. 


# 11. Final Recommendations

A teams combined RUNS has the most significant impact on their win percentage over a regular season, followed by STRIKEOUTS, then WALKS. 

Below are the direct interpretations from the statsmodel summary report, (because the data was scaled the coeffients are in each feature's standard deviation):
- For every 117.75 runs, a teams regular season win percentage increases, on average, by ~5.78%
- For every 152.97 walks, a teams regular season win percentage increases, on average, by ~2.43%
- For every 93.45 strikeout, a teams regular season win percentage _**decreases**_, on average, by ~1.83%

Converted to more standardized variables:
- For every 1 run, a teams regular season win percentage increases, on average, by ~0.0491%
- For every 1 walk, a teams regular season win percentage increases, on average, by ~0.0159%
- For every 1 strikeout, a teams regular season win percentage _**decreases**_, on average, by ~0.0195%

These findings do not shed any revolutionary??knowledge, I think??half??of Americans have seen "Moneyball", ??so inferentially it may feel a bit nonessential. However, the specificity of the coefficient translation can be helpful when comparing or narrowing down potential recruits. Furthemore, with a root mean squared error of just 5.8, this model is, on average, off by only ~5.8% in it's predictions, meaning the predictive capabilities??could prove to be useful for testing/comparing fantasy/hypothetical rosters to get some insight in how a team as a whole will perform throughout the regular season.


# 12. Next Steps


<p align="center" width="100%">
    
<img src="/images/ss_examples/IMAGE.png" alt="blue note box: DESCRIPTION">
    
</p>






























