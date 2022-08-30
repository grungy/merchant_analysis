# merchant_analysis

## Classifying Merchants

### Time Series Plotting
The first step in the data pre-processing consisted of exploring the different merchant time series. This was done by plotting randomly selected merchant time series that were resampled to 1 hour buckets. The data was resampled to provide a continuous time series between the merchant’s first transaction and their last transaction in the dataset. The resampling into hour buckets also provided some averaging to reduce the “bursti-ness” of the signal and look for general trends. Some examples of merchant time series are included in the figures below. Merchant 0x5608f200cf has the highest amount of transactions in the data set. It has a steady growth period and then a relatively stable baseline amount of transactions with a slight dip of the baseline at the end. The plot of Merchant 2bf94e2e1 has regular, but a lower density of transactions. While, Merchant 221b4736b has a high density of transactions and then periods of burstiness. In only three sample Merchants there is a high variability of patterns in the transactions.

![Alt text](./imgs/merchant_5608f200cf.png?raw=true "Transactions of Merchant 5608f200cf")

![Alt text](./imgs/merchant_2bf94e2e1.png?raw=true "Transactions of Merchant 2bf94e2e1")

![Alt text](./imgs/merchant_2bf94e2e1.png?raw=true "Transactions of Merchant 2bf94e2e1")

![Alt text](./imgs/merchant_221b4736b.png?raw=true "Transactions of merchant 221b4736b")

![Alt text](./imgs/histogram_analysis.png?raw=true "Histogram Analysis")

![Alt text](./imgs/distribution_of_transaction_sums.png?raw=true "Distribution of Transaction Sums")

### Feature Vector Development
#### Fourier Analysis
Metrics development for clustering was the most important step in determining the types of merchants on the platform. The initial hypothesis was that there would be: merchants that tried the platform and did not choose to keep using the platform, merchants who regularly use the platform, and merchants who used the platform and stopped using the platform (churn). 

Developing metrics for grouping merchants who regularly use the platform was the focus of the metrics development work. To that end a robust filtering mechanism was needed to extract these merchants from the dataset. The filtering mechanism will be discussed in detail later in the report. The metrics development had a central hypothesis that merchants will have periodic patterns based on business and human related cycles. For example, a coffee shop may post many transactions in the morning, and have steadily decreasing sales during the day. Given that we are looking for periodic trends the use of Fourier Domain analysis techniques is a good choice. It was decided to experiment with day and week trends along with their accompanying higher order harmonics.

The Fourier transform, beyond converting a merchant’s time series from the time domain to the frequency domain, provides a useful smoothing or low pass filtering of the data to remove burstiness, a phase invariant analysis of the periodicity of the time series, and elements in the feature vector that directly relate to the shape of the time series (useful for making sure the analysis makes sense). The phase invariant quality of the Fourier transform is of particular value for the platform because the business conducts business all over the world. Thus, customers will have phase shifted, but similar periodic patterns for their sales. 
Clustering Metrics Implementation

First, the dataset was loaded using the Pandas Python library. The dataset was grouped by merchant, and then sorted by number of transactions. This allowed the selection of the top 200 merchants by number of transactions. These were chosen because it was believed that they were representative of customers that regularly use the platform. 

The time series for each of the top 200 merchants was resampled into one hour buckets. The values in the buckets were summed. The resampled time series was then transformed into the Fourier domain by using Scipy's fftpack.fft function. FFT input array padding was done manually to ensure that the array was padded with the beginning of the input array rather than zeros. This made the signal more periodic and would not cause the FFT to pick up a long string of zeros as a real pattern. The code for the FFT is shown below. The output of the FFT is fed into a function called “harmonic_vector()”. 

The “harmonic_vector()” function takes the frequencies, the magnitudes returned by the FFT, and a desired frequency (fc) as input. It returns the magnitude values for fc and the four higher harmonics of fc plus or minus a tolerance percentage. For example, a day frequency was used in this analysis. harmonic_vector() would yield the magnitude of the day component of the signal, as well as, the magnitude of a half day, third day, and quarter day. 

The output of harmonic_vector() for both day and week components of the signal were then normalized to the 0th harmonic, which is the average transaction amount in this case. This was done to scale the features in a meaningful way that yielded the proportion of the revenue accounted for by daily variation. The output was used as the feature vector for the Kmeans clustering algorithm discussed later in this report. 

![Alt text](./imgs/code_1.png?raw=true "Distribution of Transaction Sums")

![Alt text](./imgs/code_2.png?raw=true "Distribution of Transaction Sums")

Looking at the plot of time series for Merchant 5e8bb6fb we can see an interesting period of inactivity in the middle of the time series. Using the Fourier analysis it is still possible to extract the needed periodicity information from the signal. The output for both day and week components are shown below. The red dots represent the magnitudes that were selected by harmonic_vector. Very strong day components, and much smaller week components can be observed.

![Alt text](./imgs/merchant_5e8bb6fb.png?raw=true "Distribution of Transaction Sums")
![Alt text](./imgs/magnitude_5e8bb6fb.png?raw=true "Distribution of Transaction Sums")
![Alt text](./imgs/magnitude2_5e8bb6fb.png?raw=true "Distribution of Transaction Sums")

### Statistical Features
Two statistical based features were also included in the feature vector to represent the magnitude of sales and transaction counts. This information was normalized out of the Fourier based features. The two features added were “average amount per transaction”, and “average transactions per day.” These two features increased the discrimination of the Kmeans plot by decreasing the number of clusters from 5 to 4 based on the output of an Elbow Plot. 

### Clustering
Clustering was performed on the feature vectors with a Kmeans clustering algorithm. An Elbow Plot was first generated to determine the number of clusters that should be used for optimal performance. The Elbow plot is included below. The point at which the plot transitions from non-linear to linear is at 4 clusters. This is why 4 clusters were chosen for fitting the Kmeans model. The cluster centers are also shown below in their entirety in one plot. However, the last two features are much larger than the other 8 because they are not normalized. This makes it hard to see what the other features look like, so a plot of the Kmeans fit on the first 8 features is also included. The first 8 features are the Fourier features and represent day, and week periodicity in the merchant’s transactions. 

Some features show a high value for the first four features and represent merchants with a high day component in their signal. Some features have a higher last four features, which represent a high week component for the merchant. The curves are distinct and should be usable as clusters for the dataset. 

![Alt text](./imgs/kmeans_cluster_centers.png?raw=true "Distribution of Transaction Sums")
![Alt text](./imgs/kmeans_cluster_centers_2.png?raw=true "Distribution of Transaction Sums")

### Results: Types of Merchants
Based on the output of the Kmeans clustering algorithm there are merchants that have a daily pattern, but not a weekly pattern that dominates their sales, merchants that have neither a strong daily pattern nor a strong weekly pattern, merchants that a weak daily pattern and a high weekly pattern, and finally merchants that have a strong daily pattern and a strong weekly pattern to their sales.

## Churn

### Definition of Churn
The definition of churn is kind of a philosophical question because there are many shades of grey. If a merchant had a high volume of transactions, but then had sporadic low value transactions is that churn? This is illustrated in the figure below where the blue and green lines are merchants that continue to have sporadic low value transactions. Another example of interest is the following figure. Where there are bursts of high transaction volume followed by long periods of no activity. This could be churn. To that end I defined churn as 2 months because it’s ⅔ of a business quarter and it seems unlikely that a business would have enough liquidity to survive a business quarter without any revenue. 
 
Churn = 2 Months

![Alt text](./imgs/cluster_4_merchants.png?raw=true "Cluster of Four Merchants")
![Alt text](./imgs/merchant_5e8bb6fb.png?raw=true "Transactions of Merchant 5e8bb6fb")

### Classifying Churn
A calculation pipeline was used to calculate whether a merchant has churned or not.

The first step was to resample the time series and calculate the counts per day of a merchant’s time series. A plot of days vs transaction counts is included below. 

![Alt text](./imgs/resampled_5e8bb6fb.png?raw=true "Transactions of Merchant 5e8bb6fb")

The next step is to threshold the signal using Otsu’s method to remove very low value transactions from being considered in this calculation. This follows my assumption that transactions that are low value compared to the values in the rest of the time series are equivalent to no transactions. This gives us a variable threshold based on maximizing the variance between two classes in a distribution. This assumes a bi-modal like distribution because there will be mode around zero and one around a high value for the actual values. The threshold chosen with Otsu’s method is shown in the red line on the histogram plot below. The threshold is used to create a binary array that represents whether there were sales or no sales in a given time period. A plot showing how the binary array lines up with the original plot of the counts.

![Alt text](./imgs/otsu_5e8bb6fb.png?raw=true "Transactions of Merchant 5e8bb6fb")
![Alt text](./imgs/binary_5e8bb6fb.png?raw=true "Transactions of Merchant 5e8bb6fb")

By finding the indexes of the binary array using numpy’s nonzero function it’s possible to take the difference of the indexes and arrive at an array that is a representation of how many days with no activity. Finally, plotting a histogram of the resulting array yields a very clear distinction between regular behavior and churn.

![Alt text](./imgs/no_sales_5e8bb6fb.png?raw=true "Transactions of Merchant 5e8bb6fb")

### Identify Churned Merchants in the Dataset
To identify churned merchants in the dataset a threshold of 60 days was chosen. Then for each merchant the last gap of days of no sales was compared to the chosen churn threshold of 60 days. If the gap of no sales exceeds the threshold the merchant is considered as churned. A full list of churned merchants is included in the zip file and called “churned_merchants.pkl”. In total, 7047 merchants were considered to have churned. This is reasonable given the distribution of the total number of transactions per merchant is heavily skewed towards the lower end of the distribution. 

This method has some weaknesses. Merchant 2d07bba was labeled as not churned, but the number of transactions is so low that it’s hard to make any conclusions about the merchant’s behavior. A more robust filtering mechanism based on activity gap and number of transactions would probably yield better results. 

![Alt text](./imgs/merchant_2d07bba.png?raw=true "Transactions of Merchant 5e8bb6fb")