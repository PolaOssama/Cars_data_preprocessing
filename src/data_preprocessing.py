import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Cars.csv')



# Convert dataset datatype to be used as a matrix

datasetNew = dataset.iloc[:,:].values


# Old way to fill missing data with imputer
'''
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)
imputer = imputer.fit(datasetNew[:, 1:3])
datasetNew[:, 1:3] = imputer.transform(datasetNew[:, 1:3])

# Rename coloumns to remove outliers 
datasetNew2 = pd.DataFrame(datasetNew,columns=['CarModel','ManufacturingYear','Cylinders','CC'	,'Price','MaximumSpeed','HorsePower','UsedNew'])
Q1 = datasetNew2.Price.quantile(0.25)
Q3 = datasetNew2.Price.quantile(0.75)
'''

# New way to fill missing data
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values = np.nan, strategy = "median")
imputer = imputer.fit(datasetNew[:,4:5])
datasetNew[:,4:5] = imputer.transform(datasetNew[:,4:5])

imputer = SimpleImputer(missing_values = np.nan, strategy = "most_frequent")
imputer = imputer.fit(datasetNew[:, 2:8])
datasetNew[:, 2:8] = imputer.transform(datasetNew[:, 2:8])


# Calculate the outlier by subtracting the third quartile from the first one 
IQR = Q3 - Q1

#print the result in terminal
IQR
# Calculate the lower limit and upper limit
lower_limit = Q1 - 1.5*IQR
upper_limit = Q3 + 1.5*IQR

# Filter the data based on the outlier so the final data will be in range between the lower limit and the upper one 
datasetNew2[(datasetNew2.Price<lower_limit)|(datasetNew2.Price>upper_limit)]

datasetFinal = datasetNew2[(datasetNew2.Price>lower_limit)&(datasetNew2.Price<upper_limit)]


# Change Object column to numerical instead of categorical with pandas get dummies 
datasetFinal = pd.get_dummies(datasetFinal, columns=['CarModel'])        
               
# Old way to convert categorical values to numerical values
'''
labelencoder = LabelEncoder()
datasetNew2[:, 0] = labelencoder.fit_transform(datasetNew2[:, 0])

onehotencoder = OneHotEncoder(categorical_features = [0])
datasetNew2 = onehotencoder.fit_transform(datasetNew2).toarray()
'''

# newer way to convert categorical values to numerical values with hotEncoder
'''
columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [0])],remainder='passthrough')
datasetNew = np.array(columnTransformer.fit_transform(datasetNew), dtype = np.float64)
'''

