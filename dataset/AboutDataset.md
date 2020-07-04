# There are two datasets :- 

Input dataset named :- Covid19_10_days_Input.csv
	this dataset does not have the last column i.e. cases_after_10_days because it has to be predicted.
	this is used for showing the prediction. and will always have 10 rows extra as compared to second dataset.
	by default the last entry in dataset will be taken as input and prediction will be done by taking this as reference
	for example at present the last row shows the data for 29/06/2020. So number of cases will be predicted for the date
	08/07/2020.

Dataset for training and testing :- DATASETtest.csv
	this dataset is used for training and testing. The number of entries in this dataset will be 10 less than 
	Covid19_10_days_Input.csv . for example if the present date is 29/06/2020. It will contain data till 
	19/06/2020.


In regression input columns are independent variables within a row, here some columns are not purely independent. Some 
columns uses the information available in the previous columns of same row in combination of an information available 
in the previous row. This was done to give some new relationship to the dataset.
