# About:
Design a customer segmmentaion model & identify customer segmments from the data understanding. This project is part of the  test for Data Scientist role at Altimetrik.

## Problem Statement:
ABC Limited company sells various products to its customers. Many customers of the company are wholesalers. Now the company has recorded all the transactions over a period of 1 year (dataset is attached with the use case). As a Machine learning Engineer, you must segment the customers in different groups so that the marketing team can push the right offers to the corresponding customers. Solve this use case using the technologies and open source tools of your choice.
The solution can include and may not be limited to:
* Exploratory analysis of data (with some visualizations if possible).
* Data pre-processing.
* Creating clusters of customers and visualizing them.
* Your Interpretation about each cluster.

## Solution
Solution contains following sections

### Data Preprocessing
1. Pre-process file to trim start and end '"', replace '""' with '"' making it ready to be read as pandas data frame.
2. Checking and removing duplicate rows.
3. Impute column 'Description': Modify product code with multiple descriptions with most occurrence product code description. Rest impute 'missing_description' for None description.
4. Impute column 'Customer_ID': Imputing Customer_ID failed with no common transaction id with and without Customer_ID. impute 'missing_cust_id_{invoice number}' for None Customer_ID making all None transaction unique and individual transactions.
5. Identify the cancellation transaction from order quantity with a negative value.
6. Calculate Amount_Spent as Quantity * Unit_Price.

### Exploratory Data Analysis/ Data Insights
1. Among all transactions around 20% of transactions are of cancellation.
2. Most of the countries have a successful transaction rate in between 65% to 85%, Chec Republic has the worst successful transaction rate. Successful Transaction Rate is defined as below:
Successful Transaction Rate = ([TotalCheckout-TotalCancellation]/[TotalCheckout]))*100
3. Most numbers of transactions are from United Kingdom followed by Germany & France
4. Most revenue is from United Kingdom followed by Netherland which has fewer unique transactions then Germany & France.
5. Most of the transactions are from the year 2011 with few from 2010.
6. A number of transactions are greater in second half of the year than the first half.
7. A number of transactions are higher towards the beginning of the month them towards the end of the month.
8. No transaction on Saturday could be weekly off on that day.
9. 12 p.m. lunchtime seems to be attracting more unique transactions & majority of transactions on a given day are between 8 a.m. to 7 p.m.
10. Number of Transaction for Products are not identical from the United Kingdom to other top 9 countries.
11. United Kindom has the most number of unique users.
12. Number of unique products having transactions in the United Kingdom is higher than that of other countries.
13. Lot of customers have transacted in between 50-200 times with the platform

### Segmmentaion Approach
Different customer clustering approaches can been tried based on available data type such as customer value data, customer transaction data, and customer purchasing sequence data.

#### Customer value data:
Customers here are clustered according to their value from purchase history. First, features are extracted from customer data, which determines how loyal customers are for the company. Model used:
* **Units price & number of units purchased**
Clustering with K-Means model with units price and frequency of the product based on customer purchases. Choosing number of cluster as 4 from Elbow method

![number of clusters](https://github.com/srikant86panda/altimetrik_ml_challenge/blob/master/image/Segmentation_Elbow.jpg)

![customer types](https://github.com/srikant86panda/altimetrik_ml_challenge/blob/master/image/Segmentation.jpg)

1. Customer Type 0 : These customers purchase few products with low price
2. Customer Type 1 : These customers purchase more products with low price
3. Customer Type 2 : These customers purchase few prodcuts from mid price range
4. Customer Type 3 : These customers purchase few products with high price

* RFM model: Used feature features recency, frequency, and monetary to cluster users based on their overall score.

![Frequency_vs_Revenue](https://github.com/srikant86panda/altimetrik_ml_challenge/blob/master/image/Frequency_vs_Revenue.jpg)

![Recency_vs_Frequency](https://github.com/srikant86panda/altimetrik_ml_challenge/blob/master/image/Recency_vs_Frequency.jpg)

![Recency_vs_Revenue](https://github.com/srikant86panda/altimetrik_ml_challenge/blob/master/image/Recency_vs_Revenue.jpg)

1. Customer Type High-Value : These customers are frequent and monetary is high or recent with high frequency and monetary
2. Customer Type Mid-Value : These customers are frequent with mid monetary and seen recent
3. Customer Type Low-Value : These customers are frequent and monetary is high

#### Customer transaction data:
Using transaction data revingeal customer preferences and needs using doc2vec approach to model user top prefered prodcut description.
Basically this approach can segment user in to cluster with similar kind of purchase preference from history.
**User Embedding**
![doc2vec_user_embedding](https://github.com/srikant86panda/altimetrik_ml_challenge/blob/master/image/doc2vec_embedding.jpg)

e.g. of user segmented:
![doc2vec_user_embedding](https://github.com/srikant86panda/altimetrik_ml_challenge/blob/master/image/doc2vec_segments.jpg)

#### Customer purchasing sequence data:
Pending
