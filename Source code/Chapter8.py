

#Load libraries
import numpy as np
from sklearn import preprocessing

# Create feature
feature = np.array([[-500.5],
                    [-100.1],
                    [0],
                    [100.1],
                    [900.9]])

# Create scaler
minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))

# Scale feature
scaled_feature = minmax_scale.fit_transform(feature)

# Show feature
print(scaled_feature)

'''
[[0.        ]
 [0.28571429]
 [0.35714286]
 [0.42857143]
 [1.        ]]
'''





# Load libraries
import numpy as np
from sklearn import preprocessing

# Create feature
x = np.array([[-1000.1],
                [-200.2],
                [500.5],
                [600.6],
                [9000.9]])

# Create scaler
scaler = preprocessing.StandardScaler()

# Transform the feature
standardized = scaler.fit_transform(x)

# Show feature
print(standardized)

'''
[[-0.76058269]
 [-0.54177196]
 [-0.35009716]
 [-0.32271504]
 [ 1.97516685]]
'''



# Print mean and standard deviation
print("Mean:", round(standardized.mean()))
print("Standard deviation:", standardized.std())

'''
Mean: 0.0
Standard deviation: 1.0
'''







# Load libraries
import numpy as np
from sklearn import preprocessing

# Create feature
x = np.array([[-1000.1],
                [-200.2],
                [500.5],
                [600.6],
                [9000.9]])

# Create scaler
robust_scaler = preprocessing.RobustScaler()

# Transform feature
x = robust_scaler.fit_transform(x)

# Show result
print(x)

'''
[[-1.87387612]
 [-0.875     ]
 [ 0.        ]
 [ 0.125     ]
 [10.61488511]]
'''





# Load libraries
import numpy as np
from sklearn.preprocessing import Normalizer

# Create feature matrix
features = np.array([[0.5, 0.5],
                     [1.1, 3.4],
                     [1.5, 20.2],
                     [1.63, 34.4],
                     [10.9, 3.3]])

# Create normalizer
normalizer = Normalizer(norm="l2")

# Transform feature matrix
x = normalizer.transform(features)

# Show result
print(x)

'''
[[0.70710678 0.70710678]
 [0.30782029 0.95144452]
 [0.07405353 0.99725427]
 [0.04733062 0.99887928]
 [0.95709822 0.28976368]]
'''



# Transform feature matrix
features_l2_norm = Normalizer(norm="l2").transform(features)

# Show feature matrix
x2 = features_l2_norm

# Show result
print(x2)

'''
[[0.70710678 0.70710678]
 [0.30782029 0.95144452]
 [0.07405353 0.99725427]
 [0.04733062 0.99887928]
 [0.95709822 0.28976368]]
'''


# Transform feature matrix
features_l1_norm = Normalizer(norm="l1").transform(features)

# Show feature matrix
x3 =features_l1_norm

# Show result
print(x3)

'''
[[0.5        0.5       ]
 [0.24444444 0.75555556]
 [0.06912442 0.93087558]
 [0.04524008 0.95475992]
 [0.76760563 0.23239437]]
'''



# Print sum
print("Sum of the first observation\'s values:",
features_l1_norm[0, 0] + features_l1_norm[0, 1])

'''
Sum of the first observation's values: 1.0
'''




# Load libraries
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

# Create feature matrix
features = np.array([[2, 3],
[2, 3],
[2, 3]])

# Create PolynomialFeatures object
polynomial_interaction = PolynomialFeatures(degree=2, include_bias=False)

# Create polynomial features
x = polynomial_interaction.fit_transform(features)

# Show Result
print(x)

'''
    [[2. 3. 4. 6. 9.]
     [2. 3. 4. 6. 9.]
     [2. 3. 4. 6. 9.]]
'''




interaction = PolynomialFeatures(degree=2,
              interaction_only=True, include_bias=False)
x2 = interaction.fit_transform(features)
print(x2)
'''
    [[2. 3. 6.]
     [2. 3. 6.]
     [2. 3. 6.]]
'''





# Load libraries
import numpy as np
from sklearn.preprocessing import FunctionTransformer

# Create feature matrix
features = np.array([[2, 3],
[2, 3],
[2, 3]])

# Define a simple function
def add_ten(x):
    return x + 10

# Create transformer
ten_transformer = FunctionTransformer(add_ten)

# Transform feature matrix
x = ten_transformer.transform(features)

# Show Result
print(x)

'''
    [[12 13]
     [12 13]
     [12 13]]
'''




# Load library
import numpy as np
import pandas as pd

# Create feature matrix
features1 = np.array([[2, 3],
                      [2, 3],
                      [2, 3]])

features2 = np.array([[2, 3],
                      [2, 3],
                      [2, 3]])

# Define a simple function
def add_ten(x):
    return x + 10

# Create DataFrame
df = pd.DataFrame(features1, columns=["feature1", "feature2"])

# Apply function
x = df.apply(add_ten)

# Show result
print(x)

'''
       feature1  feature2
    0        12        13
    1        12        13
    2        12        13
'''








# Load libraries
import numpy as np
from sklearn.covariance import EllipticEnvelope
from sklearn.datasets import make_blobs

# Create simulated data
features, _ = make_blobs(n_samples = 10,
                n_features = 2,
                centers = 1,
                random_state = 1)

# Replace the first observation's values with extreme values
features[0,0] = 10000
features[0,1] = 10000

# Create detector
outlier_detector = EllipticEnvelope(contamination=.1)

# Fit detector
outlier_detector.fit(features)

# Predict outliers
x = outlier_detector.predict(features)

# Show result
print(x)

'''
    [-1  1  1  1  1  1  1  1  1  1]
'''





# Create one feature
feature = features[:,0]

# Create a function to return index of outliers
def indicies_of_outliers(x):
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - (iqr * 1.5)
    upper_bound = q3 + (iqr * 1.5)
    return np.where((x > upper_bound) | (x < lower_bound))

# Run function
x2 = indicies_of_outliers(feature)

# Show result
print(x2)

'''
(array([0], dtype=int64),)
'''





# Load library
import pandas as pd

# Create DataFrame
houses = pd.DataFrame()

# Create rows of data
houses['Price'] = [534433, 392333, 293222, 4322032]
houses['Bathrooms'] = [2, 3.5, 2, 116]
houses['Square_Feet'] = [1500, 2500, 1500, 48000]

# Filter observations
x = houses[houses['Bathrooms'] < 20]

# Show result
print('\nOriginal dataframe\n----------------')
print(houses)
print('\nResult\n----------------')
print(x)

'''
    Original dataframe
    ----------------
         Price  Bathrooms  Square_Feet
    0   534433        2.0         1500
    1   392333        3.5         2500
    2   293222        2.0         1500
    3  4322032      116.0        48000
    
    Result
    ----------------
        Price  Bathrooms  Square_Feet
    0  534433        2.0         1500
    1  392333        3.5         2500
    2  293222        2.0         1500
'''




# Load library
import pandas as pd
import numpy as np

# Create DataFrame
houses = pd.DataFrame()

# Create rows of data
houses['Price'] = [534433, 392333, 293222, 4322032]
houses['Bathrooms'] = [2, 3.5, 2, 116]
houses['Square_Feet'] = [1500, 2500, 1500, 48000]

# Create feature based on boolean condition
houses["Outlier"] = np.where(houses["Bathrooms"] < 20, 0, 1)

# Show result
print(houses)

'''
         Price  Bathrooms  Square_Feet  Outlier
    0   534433        2.0         1500        0
    1   392333        3.5         2500        0
    2   293222        2.0         1500        0
    3  4322032      116.0        48000        1
'''



# Load library
import pandas as pd
import numpy as np

# Create DataFrame
houses = pd.DataFrame()

# Create rows of data
houses['Price'] = [534433, 392333, 293222, 4322032]
houses['Bathrooms'] = [2, 3.5, 2, 116]
houses['Square_Feet'] = [1500, 2500, 1500, 48000]

# Log feature
houses["Log_Of_Square_Feet"] = [np.log(x) for x in houses["Square_Feet"]]

# Show result
print(houses)

'''
         Price  Bathrooms  Square_Feet  Log_Of_Square_Feet
    0   534433        2.0         1500            7.313220
    1   392333        3.5         2500            7.824046
    2   293222        2.0         1500            7.313220
    3  4322032      116.0        48000           10.778956
'''












# Load libraries
import numpy as np
from sklearn.preprocessing import Binarizer

# Create feature
age = np.array([[6],
               [12],
               [20],
               [36],
               [65]])

# Create binarizer
binarizer = Binarizer(18)

# Transform feature
x = binarizer.fit_transform(age)

# Show Result
print(x)

'''
        [[0]
         [0]
         [1]
         [1]
         [1]]
'''






# Bin feature
x2 = np.digitize(age, bins=[20,30,64])

# Show Result
print(x2)

'''
        [[0]
         [0]
         [1]
         [2]
         [3]]
'''






# Bin feature
x3 = np.digitize(age, bins=[20,30,64], right=True)

# Show Result
print(x3)

'''
    [[0]
     [0]
     [0]
     [2]
     [3]]
'''



# Bin feature
x4 = np.digitize(age, bins=[18])

# Show Result
print(x4)

'''
    [[0]
     [0]
     [1]
     [1]
     [1]]
'''




# Load libraries
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Make simulated feature matrix
features, _ = make_blobs(n_samples = 50,
                         n_features = 2,
                         centers = 3,
                         random_state = 1)

# Create DataFrame
dataframe = pd.DataFrame(features, columns=["feature_1", "feature_2"])

# Make k-means clusterer
clusterer = KMeans(3, random_state=0)

# Fit clusterer
clusterer.fit(features)

# Predict values
dataframe["group"] = clusterer.predict(features)

# View first few observations
print(dataframe.head(5))

'''
       feature_1  feature_2  group
    0  -9.877554  -3.336145      2
    1  -7.287210  -8.353986      0
    2  -6.943061  -7.023744      0
    3  -7.440167  -8.791959      0
    4  -6.641388  -8.075888      0
'''





# Load library
import numpy as np

# Create feature matrix
features = np.array([[1.1, 11.1],
                     [2.2, 22.2],
                     [3.3, 33.3],
                     [4.4, 44.4],
                     [np.nan, 55]])

# Keep only observations that are not (denoted by ~) missing
x = features[~np.isnan(features).any(axis=1)]

# Show result
print(x)

'''
    [[ 1.1 11.1]
     [ 2.2 22.2]
     [ 3.3 33.3]
     [ 4.4 44.4]]
'''





# Load library
import numpy as np
import pandas as pd

# Create feature matrix
features = np.array([[1.1, 11.1],
                     [2.2, 22.2],
                     [np.nan, 33.3],
                     [4.4, 44.4],
                     [5.2, 55]])

# Load data
dataframe = pd.DataFrame(features, columns=["feature1", "feature2"])

# Remove observations with missing values
x = dataframe.dropna()

# Show result
print('\nOriginal data\n--------------------')
print(dataframe)
print('\nResult\n--------------------')
print(x)

'''
Original data
--------------------
   feature1  feature2
0       1.1      11.1
1       2.2      22.2
2       NaN      33.3
3       4.4      44.4
4       5.2      55.0

Result
--------------------
   feature1  feature2
0       1.1      11.1
1       2.2      22.2
3       4.4      44.4
4       5.2      55.0
'''





from sklearn.impute import KNNImputer
import numpy as np
import pandas as pd

dict = {'First': [100, 90, np.nan, 95],
        'Second': [30, 45, 56, np.nan],
        'Third': [np.nan, 40, 80, 98]}

# creating a dataframe from list
df = pd.DataFrame(dict)

# Initialize KNNImputer
imputer = KNNImputer(n_neighbors=2)

# Impute/Fill Missing Values
x = imputer.fit_transform(df)

# Show result
print('\nOriginal data\n--------------------')
print(df)
print('\nResult\n--------------------')
print(x)

'''
Original data
--------------------
   First  Second  Third
0  100.0    30.0    NaN
1   90.0    45.0   40.0
2    NaN    56.0   80.0
3   95.0     NaN   98.0

Result
--------------------
[[100.   30.   69. ]
 [ 90.   45.   40. ]
 [ 97.5  56.   80. ]
 [ 95.   43.   98. ]]
'''




# Load library
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer, MissingIndicator

# Make a simulated feature matrix
features, _ = make_blobs(n_samples = 1000,
                        n_features = 2,
                        random_state = 1)

# Standardize the features
scaler = StandardScaler()
standardized_features = scaler.fit_transform(features)

# Replace the first feature's first value with a missing value
true_value = standardized_features[0,0]
standardized_features[0,0] = np.nan

# Create imputer
mean_imputer = SimpleImputer(strategy="mean")

# Impute values
features_mean_imputed = mean_imputer.fit_transform(features)

# Compare true and imputed values
print("True Value:", true_value)
print("Imputed Value:", features_mean_imputed[0,0])

'''
True Value: 0.8730186113995938
Imputed Value: -3.058372724614996
'''
