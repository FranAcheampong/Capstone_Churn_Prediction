#  Expressor Customer Churn Prediction, ML Approach
<p align="center">
  <img src="images/welcome.jpg" alt="ExpressorLogo" width="800">
</p>

## Project Overview
In this project, we aim to find the likelihood of a customer leaving the organization, the key indicators of churn as well as the retention strategies that can be implemented to avert this problem.
Churn is a one of the biggest problem in the telecom industry. Research has shown that the average monthly churn rate among the top 4 wireless carriers in the US is 1.9% - 2%

Customer attrition is one of the biggest expenditures of any organization. Customer churn otherwise known as customer attrition or customer turnover is the percentage of customers that stopped using your company's product or service within a specified timeframe. For instance, if you began the year with 500 customers but later ended with 480 customers, the percentage of customers that left would be 4%. If we could figure out why a customer leaves and when they leave with reasonable accuracy, it would immensely help the organization to strategize their retention initiatives manifold.


<p align="center">
  <img src="images/logo.jpg" alt="ExpressorLogo" width="800">
</p>


This solution will help this telecom company to better serve their customers by understanding which customers are at risk of leaving.


## The presentation follows the following outline

- [Project Overview](#project-overview)
- [Getting Started](#getting-started)
- [Data](#data)
- [Modeling](#modeling)
- [Evaluation](#evaluation)
- [Deployment](#deployment)


## Objectives

- Objective 1: Data Exploration
- Objective 2: Data Preprocessing
- Objective 3: Model Selection and Training
- Objective 4: Model Evaluation
- Objective 5: Results & Analysis
- Objective 6: Deployment and Future Improvements


## Summary
| Code | Name                                                | Summary of the work                                                                                          |                                                                                              Streamlit App    |                                                                                                |
|------|-----------------------------------------------------|------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|
| Capstone  | Expressor Customer Churn Prediction, ML Approach     | [Summary_PPT](https://tinyurl.com/5bpuasbc) |  [Streamlit App](https://huggingface.co/spaces/HOLYBOY/Customer_Churn_App)      |



## Project Setup

To set up the project environment, follow these steps:

1. Clone the repository:

git clone my_github 

```bash 
https://github.com/FranAcheampong/Capstone_Churn_Prediction.git
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Create a virtual environment:

- **Windows:**
  ```bash
  python -m venv venv
  venv\Scripts\activate
  ```

You can copy each command above and run them in your terminal to easily set up the project environment.


## Data

The data set used in this project was sourced from the [Zindi](https://zindi.africa/competitions/customer-churn-prediction-challenge-for-azubian).

## Data set Description

| Column Name     | Type   | Description                                                              |
|-----------------|-----------------|--------------------------------------------------------------------------|
| REGION          | Categorical     | The location of each client                                               |
| TENURE          | Numeric         | Duration with the network                                                 |
| MONTANT         | Numeric         | Top-Up Amount                                                             |
| FREQUENCE_RECH  | Numeric         | The number of times a customer refilled                                    |
| REVENUE         | Numeric         | Monthly income of each client                                             |
| ARPU_SEGMENT    | Numeric         | Income over 90 days divided by 3                                          |
| FREQUENCE       | Numeric         | Number of times the client has made an income                              |
| DATA_VOLUME     | Numeric         | Number of connections                                                     |
| ON_NET          | Numeric         | Inter Expresso call                                                       |
| ORANGE          | Numeric         | Calls to Orange                                                           |
| TIGO            | Numeric         | Calls to Tigo                                                             |
| ZONE1           | Numeric         | Calls to Zone1                                                            |
| ZONE2           | Numeric         | Calls to Zone2                                                            |
| MRG             | Categorical     | A client who is going                                                      |
| REGULARITY      | Numeric         | Number of times the client is active for 90 days                           |
| TOP_PACK        | Categorical     | The most active packs                                                     |
| FREQ_TOP_PACK   | Numeric         | Number of times the client has activated the top pack packages             |
| CHURN           | Binary          | Target variable to predict - Churn (Positive: customer will churn, Negative: customer will not churn) |


## Exploratory Data Analysis

During the exploratory data analysis (EDA) phase, a comprehensive investigation of the churn dataset was conducted to gain insights through various types of analyses.

- **Univariate analysis:** A thorough examination of each variable individually was performed. Summary statistics such as mean, median, standard deviation, and quartiles were calculated to understand the central tendency and spread of the data.

<p align="center">
  <img src="images/Univariate.png" alt="Univariate" width="600">
</p>

- **Bivariate analysis:** Relationships between pairs of variables were explored to identify patterns and potential predictor variables for sepsis classification.

<p align="center">
  <img src="images/Bivariate.png" alt="Bivariate" width="600">
</p>

- **Multivariate analysis:** Relationships among multiple variables were examined simultaneously, allowing for a deeper understanding of their interactions and impact on sepsis.

<p align="center">
  <img src="images/multivariate.png" alt="multivariate" width="600">
</p>

### Hypotheses:
These hypotheses, along with the results of the EDA, contribute to a deeper understanding of the dataset and provide valuable insights for further analysis and model development.

1. Customers with longer tenure are less likely to churn than those with short tenure.
Null: Customers with longer tenure are less likely to churn
Alternative: Customers with shorter tenure are more likely to churn

<p align="center">
  <img src="images/Hypothesis1.png" alt="H1" width="600">
</p>
With the over all churn rate of 18.8%, Mid-term tenure contracts or subscriber turn to churn more than all other type of contracts. This is followed by medium-term tenures. Whiles the very short-term contract turn to churn the least. Further investigations needs to be conducted to obtain the right insight because earlier analysis showed that very short-term tenure had very small subscribers or users.


2. Customers with lesser income are likely to churn than those who have higher
Null: Customer with less income are likely to churn
Alternative: Customers with a bigger income are likely to churn
<p align="center">
  <img src="images/Hypothesis2.png" alt="H2" width="600">
</p>

Based on the results, there is evidence to reject the null hypothesis. The difference in churn rates between the lower income group and the higher income group is statistically significant.

### Business Questions
1. What type of services offered by the telecom industry
<p align="center">
  <img src="images/Q1.png" alt="Question1" width="600">
</p>
The bar chart presentated above indicates that voice related service has the greater share of the dataset. It is followed by data related service, with messaging subscribers been the least of the dataset. Expressor management must first conduct surveys and gather feedback loops to determine why users are patronizing one package than the other

2. What is the overall churn rate for the telecom company during the observed period?
<p align="center">
  <img src="images/Q2.png" alt="Question2" width="600">
</p>
Based on the pie above it can be concluded that majority of the subscriber dont churn (81.2%). With the average churn rate of 18.8% of the subscribers. Even though the dataset is highly inbalanced, the churn rate is higher than the industry standard of 1.5-2%. Expressor management needs to urgently take measures to curtails this problem.

3. Are there any specific regions or geographic areas with a higher churn rate compared to others?
<p align="center">
  <img src="images/Q3.png" alt="Question3" width="600">
</p>
The bar plot displays the average churn rates for all the regions.

From the graph, it can be seen that DAKAR is the only region with average churn rate greater than the overall churn rate.
SEDHIOU is the second highest in term of churn rate as has 5.4% of churn.
All the other regions are lower than the overall average churn rate.

4. What is the relationship between tunure and churn?
<p align="center">
  <img src="images/Q4.png" alt="Question4" width="600">
</p>
Long term and very short term term tenures are the two tenures with their churn rate lower than the overall churn rate.
Medium term, Midterm and short term have churn rate above the overall churn rate.
This means that Medium to short term tenures turn to churn more
5. Are customers who have a higher number of on-net calls (ON_NET) less likely to churn?
<p align="center">
  <img src="images/Q5.png" alt="Question5" width="600">
</p>
The bar plot, put the on net call in to categories starting from 0-100, 101-200, 201-300, 301-400, 401-500 and infinally 500+ and calculates the the numbers of times they make calls in that range. The following are the observations made

201-300, had the majority of calls followed by 0-100
the other ranges had less than 2%

## Modeling

During the modeling phase, the evaluation of models took into consideration the imbalanced nature of the data. The best performance evaluation estimator would be the AUC score, which provide a balanced assessment for imbalanced datasets.

We trained the underlisted six models and evaluated their performance based on Area Under the Curve (AUC)

- **Logistic Regression** 
- **Decision Tree** 
- **Random Forest** 
- **GaussianNB**
- **ComplementNB**
- **Support Vector Machine (SVM)**

These models were evaluated based on their AUC and logloss scores, providing insights into their performance on the imbalanced dataset. 


Given the imbalanced nature of our dataset, we assessed the models' performance using the AUC metric.

- Logistic Regression model emerged as th top-performing model, achieving the highest AUC scores of 80%.
- ComplementNB consistently demonstrated high performance across different conditions.
- GaussianNB had a relatively lower AUC score and higher log loss compared to other models.


### Streamlit deployment 

Navigate to the cloned repository and run the command:

```bash 
pip install -r requirements.txt
``` 
To run the demo app (being at the repository root), use the following command:
```bash 
streamlit run StreamlitApp.py
```

### App Execution on Huggingface

Here's a step-by-step process on how to use the [Streamlit App](https://huggingface.co/spaces/HOLYBOY/Customer_Churn_App) on Huggingface:



## Contribution
You contribution, critism etc are welcome. We are willing to colaborate with any data analyst/scientist to improve this project. Thank your 

## Contact

`1. ACHEAMPONG Francis` 

`Data Analyst`
`Azubi Africa`

- [![LinkedIn](https://img.shields.io/badge/LinkedIn-%230077B5?logo=linkedin&logoColor=orange)](https://www.linkedin.com/in/francis-acheampong)

`2. Kehinde Afolabi` 

`Data Analyst`
`Azubi Africa`

- [![LinkedIn](https://img.shields.io/badge/LinkedIn-%230077B5?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/kehinde-afolabi-a14572b1)

`3. Rekia Ouedraogo` 

`Data Analyst`
`Azubi Africa`

- [![LinkedIn](https://img.shields.io/badge/LinkedIn-%230077B5?logo=linkedin&logoColor=blue)](https://www.linkedin.com/in/rekia-iddrisu-ouedraogo)

`4. Stephen Tetteh Okoe` 

`Data Analyst`
`Azubi Africa`

- [![LinkedIn](https://img.shields.io/badge/LinkedIn-%230077B5?logo=linkedin&logoColor=pink)](https://www.linkedin.com/in/stephen-tetteh-okoe-849b34244)

`5. Mike Kerich` 

`Data Analyst`
`Azubi Africa`

- [![LinkedIn](https://img.shields.io/badge/LinkedIn-%230077B5?logo=linkedin&logoColor=yellow)](https://www.linkedin.com/in/mike-kerich-6952081a0)

