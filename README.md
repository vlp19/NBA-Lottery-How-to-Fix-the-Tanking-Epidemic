# NBA Tanking Prevention: A Neural Network & Monte Carlo Simulation Approach
## Project Overview:
This project addresses the critical issue of tanking in the NBA. It aims to give a general solution to what has hitherto been a race to last place for many suffering teams.
Methodology
Our solution integrates a multi-pronged analytical pipeline:

### Data Collection & Feature Engineering: 
Leveraging the nba_api and PostgreSQL, I compiled historical team performance data, including wins, losses, playoff success, 
and advanced metrics like Defensive Rebound Percentage (DREB) and Defensive Rating (DRTG). We engineered key features such as a 'Success Score,' 
a 'Performance Anomaly Index (PAI)' to capture deviations from historical performance and an 'Effort' metric, derived from DREB and DRTG, which serves as an injury buffer.

### Neural Network for Tanking Classification: 
A custom-built PyTorch neural network (TankingClassifier) was trained on a curated dataset of historical 'tanking' 
and 'non-tanking' seasons. The network utilizes the PAI, win percentage, average historical success, and the 'Effort' metric as inputs to predict a 'Tanking Probability' 
($T_p$) between 0 and 1.

### Monte Carlo Simulation for Policy Evaluation: 
We developed a Monte Carlo simulation engine to model the NBA Draft Lottery under various policy interventions. 
This engine takes the $T_p$ values from the neural network and simulates different 'tax levels' (e.g., Gentle, Moderate, Aggressive). 
Each tax level penalizes tanking teams following this heirarchy:
- #### Gentle:
A 25% tax. Tanking teams have their original odds for the top four picks multiplied by $\frac{3}{4}$
- #### Moderate:
A 50% tax. Tanking teams have their original odds for the top four picks multiplied by $\frac{1}{2}$
- #### Aggressive:
A 100% tax. Tanking teams lose all chances of getting any of the top four picks

When a team is identified as a high-probability tanker, a percentage of its top-pick lottery odds is 'taxed' and proportionally redistributed among non-tanking teams. 
The simulation runs thousands of virtual drafts to statistically determine the impact on first-pick probabilities across all lottery seeds.

## Key Findings & Impact
The Monte Carlo simulations yielded compelling insights:

### Flattening Distribution: 
As the 'tax level' on tanking teams becomes more aggressive, the distribution of the first pick's probability across lottery seeds flattens. 
This effectively diminishes the advantage of holding the absolute worst record, directly disincentivizing the 'race to the bottom.'
### Beneficial Redistribution: 
Non-tanking teams, particularly those with mid-to-lower lottery seeds (e.g., 6th seed and beyond), see an increased probability 
of securing the first pick under aggressive tax levels. This supports legitimate rebuilding efforts without penalizing teams for genuine struggles.

### Optimal Policy Insights: 
The project provides a quantitative framework for the NBA to choose an optimal tax level that balances 
discouraging tanking with maintaining fair competitive balance. The results suggest that 'Moderate' to 'Aggressive' tax rates are highly effective.

## Technologies Used
Postgresql: Data wrangling
Python: Core programming language
pandas: Data manipulation and analysis
numpy: Numerical operations
nba_api: NBA statistics data retrieval
psycopg2: PostgreSQL database interaction
torch: Neural network development and training
seaborn, matplotlib: Data visualization

## Conclusion
This project demonstrates a robust, data-driven methodology to combat NBA tanking. 
Using the combination of behavioral classification in machine learning for a binary classifier neural network and policy testing with monte carlo simulations,
we explore how a "tax and redistribute" model to effectively penalize tanking while maintaining fairness. It is found that 

