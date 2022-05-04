import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Cars.csv')



# Convert dataset datatype to be used as a matrix

datasetNew = dataset.iloc[:,:].values


# Fill missing data
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values = np.nan, strategy = "median")
imputer = imputer.fit(datasetNew[:,4:5])
datasetNew[:,4:5] = imputer.transform(datasetNew[:,4:5])

imputer = SimpleImputer(missing_values = np.nan, strategy = "most_frequent")
imputer = imputer.fit(datasetNew[:, 2:8])
datasetNew[:, 2:8] = imputer.transform(datasetNew[:, 2:8])


# Rename coloumns to remove outliers 
datasetNew2 = pd.DataFrame(datasetNew,columns=['CarModel','ManufacturingYear','Cylinders','CC'	,'Price','MaximumSpeed','HorsePower','UsedNew'])
Q1 = datasetNew2.Price.quantile(0.25)
Q3 = datasetNew2.Price.quantile(0.75)

IQR = Q3 - Q1
IQR

lower_limit = Q1 - 1.5*IQR
upper_limit = Q3 + 1.5*IQR

datasetNew2[(datasetNew2.Price<lower_limit)|(datasetNew2.Price>upper_limit)]


datasetFinal = datasetNew2[(datasetNew2.Price>lower_limit)&(datasetNew2.Price<upper_limit)]


# Change Object column to numerical instead of categorical
datasetFinal = pd.get_dummies(datasetFinal, columns=['CarModel'])        
               
