# Term Deposit Marketing Analysis - Key Findings

## 1. Data Overview Findings
- The dataset contains 40,000 records with 14 columns
- No missing values or duplicated entries were found in the dataset
- The target variable shows that only 7.24% of customers subscribed to term deposits
- The data includes demographic information (age, job, marital status, education), financial indicators (balance, housing/personal loans), and campaign details (contact method, duration, etc.)

## 2. Preprocessing Findings
- Data quality is high with no missing values or duplicates
- The target variable was converted from categorical ("yes"/"no") to binary (1/0) format for analysis
- No feature engineering was needed beyond creating an age group variable for segmentation

## 3. Univariate Analysis Findings

### 3.1 Numerical Features
- **Age**: Average customer age is 40.5 years, with most customers between 33-48 years
- **Balance**: Average yearly balance is €1,274, but with high variability (std dev: €2,903)
- **Duration**: Call duration averages 254 seconds, with high variability
- **Campaign**: Most customers were contacted only 1-3 times during the campaign

### 3.2 Categorical Features
- **Job**: Blue-collar workers (23.5%), management (20.4%), and technicians (17.1%) are the most common occupations
- **Marital Status**: Married customers represent the majority (61%)
- **Education**: Secondary education is most common (52.5%), followed by tertiary (28%)
- **Loans**: 60% have housing loans, while only 17.3% have personal loans
- **Contact**: Most customers were contacted via cellular phone (62.3%)
- **Month**: May (33.8%), July (16%), and August (13%) were the most active campaign months

## 4. Bivariate Analysis Findings

### 4.1 Numerical Features vs. Target
- **Duration**: Strongest correlation with subscription (0.46) - longer calls are associated with higher subscription rates
- **Campaign**: Negative correlation (-0.04) - more contact attempts are associated with lower subscription rates
- **Balance**: Slight positive correlation (0.03) - higher balances are weakly associated with higher subscription rates
- **Age**: Slight negative correlation (-0.02) - younger customers are slightly more likely to subscribe

### 4.2 Categorical Features vs. Target
- **Job**: Students (15.7%), retired (10.5%), and unemployed (8.7%) have the highest subscription rates
- **Marital Status**: Single customers have the highest subscription rate (9.4%)
- **Education**: Tertiary education has the highest subscription rate (9.2%)
- **Contact**: Cellular contact method yields the highest subscription rate (9%)
- **Month**: October (61.3%) and March (48.5%) show dramatically higher subscription rates
- **Housing/Loans**: Customers without housing loans or personal loans are more likely to subscribe

## 5. Segment Analysis Findings
- **Age Groups**: 
  - Youngest (≤29) and oldest (60+) age groups have the highest subscription rates (10.3% and 38.9% respectively)
  - Middle-aged customers (30-59) have lower subscription rates (6-7%)
- **High-Value Segments**:
  - Students with tertiary education
  - Retired customers with high balances
  - Single customers without loans
  - Customers contacted in March and October

## 6. Visualization Findings
- Age distribution shows a right-skewed pattern with most customers in the 30-45 range
- Balance distribution is highly skewed with most customers having relatively low balances
- Call duration shows a right-skewed distribution with most calls under 300 seconds
- Top jobs by subscription rate (students, retired, unemployed) differ from the most common jobs in the dataset
- Correlation heatmap confirms that call duration is the strongest predictor of subscription

## 7. Overall Key Insights
1. **Call Duration Impact**: Longer call duration strongly indicates interest and higher likelihood of subscription
2. **Seasonal Effects**: Certain months (October, March) show dramatically higher success rates
3. **Demographic Patterns**: Students, retired persons, and those with higher education are more receptive
4. **Financial Factors**: Customers without existing loans and with higher balances are more likely to subscribe
5. **Contact Method Matters**: Cellular contact yields better results than other methods
6. **Campaign Fatigue**: Multiple contact attempts correlate with lower subscription rates

## 8. Recommendations for Marketing Strategy
1. **Target High-Value Segments**: Focus on students, retired persons, and customers without existing loans
2. **Optimize Timing**: Prioritize campaigns in October and March when conversion rates are highest
3. **Improve Call Quality**: Since longer calls correlate with subscriptions, train staff to engage customers effectively
4. **Contact Method**: Prioritize cellular contact over other methods
5. **Avoid Over-Contact**: Limit the number of contact attempts per customer to prevent campaign fatigue
6. **Personalized Approach**: Develop tailored messaging for different age groups, especially for the youngest and oldest segments
7. **Education Focus**: Create materials that appeal to higher-educated customers who show greater interest
