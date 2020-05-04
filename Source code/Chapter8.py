Load libraries
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
scaled_feature
array([[ 0. ],
[ 0.28571429],
[ 0.35714286],
[ 0.42857143],
[ 1. ]])


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
standardized
array([[-0.76058269],
[-0.54177196],
[-0.35009716],
[-0.32271504],
[ 1.97516685]])



Print mean and standard deviation
print("Mean:", round(standardized.mean()))
print("Standard deviation:", standardized.std())



# Create scaler
robust_scaler = preprocessing.RobustScaler()
# Transform feature
robust_scaler.fit_transform(x)



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
normalizer.transform(features)
array([[ 0.70710678, 0.70710678],
[ 0.30782029, 0.95144452],
[ 0.07405353, 0.99725427],
[ 0.04733062, 0.99887928],
[ 0.95709822, 0.28976368]])



# Transform feature matrix
features_l2_norm = Normalizer(norm="l2").transform(features)
# Show feature matrix
features_l2_norm
array([[ 0.70710678, 0.70710678],
[ 0.30782029, 0.95144452],
[ 0.07405353, 0.99725427],
[ 0.04733062, 0.99887928],
[ 0.95709822, 0.28976368]])


# Transform feature matrix
features_l1_norm = Normalizer(norm="l1").transform(features)
# Show feature matrix
features_l1_norm
array([[ 0.5 , 0.5 ],
[ 0.24444444, 0.75555556],
[ 0.06912442, 0.93087558],
[ 0.04524008, 0.95475992],
[ 0.76760563, 0.23239437]])


# Print sum
print("Sum of the first observation\'s values:",
features_l1_norm[0, 0] + features_l1_norm[0, 1])
Sum



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
polynomial_interaction.fit_transform(features)
array([[ 2., 3., 4., 6., 9.],
[ 2., 3., 4., 6., 9.],
[ 2., 3., 4., 6., 9.]])



interaction = PolynomialFeatures(degree=2,
interaction_only=True, include_bias=False)
interaction.fit_transform(features)
array([[ 2., 3., 6.],
[ 2., 3., 6.],
[ 2., 3., 6.]])



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
ten_transformer.transform(features)
array([[12, 13],
[12, 13],
[12, 13]])

# Load library
import pandas as pd
# Create DataFrame
df = pd.DataFrame(features, columns=["feature_1", "feature_2"])
68 | Chapter 4: Handling Numerical Data
# Apply function
df.apply(add_ten)


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
outlier_detector.predict(features)
array([-1, 1, 1, 1, 1, 1, 1, 1, 1, 1])


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
indicies_of_outliers(feature)
(array([0]),)



# Load library
import pandas as pd
# Create DataFrame
houses = pd.DataFrame()
houses['Price'] = [534433, 392333, 293222, 4322032]
houses['Bathrooms'] = [2, 3.5, 2, 116]
houses['Square_Feet'] = [1500, 2500, 1500, 48000]
# Filter observations
houses[houses['Bathrooms'] < 20]




# Load library
import numpy as np
# Create feature based on boolean condition
houses["Outlier"] = np.where(houses["Bathrooms"] < 20, 0, 1)
# Show data
houses


# Log feature
houses["Log_Of_Square_Feet"] = [np.log(x) for x in houses["Square_Feet"]]
# Show data
houses



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
binarizer.fit_transform(age)
array([[0],
[0],
[1],
[1],
[1]])

# Bin feature
np.digitize(age, bins=[20,30,64])
array([[0],
[0],
[1],
[2],
[3]])


# Bin feature
np.digitize(age, bins=[20,30,64], right=True)
array([[0],
[0],
[0],
[2],
[3]])


Bin feature
np.digitize(age, bins=[18])
array([[0],
[0],
[1],
[1],
[1]])




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
dataframe.head(5)



# Load library
import numpy as np
# Create feature matrix
features = np.array([[1.1, 11.1],
[2.2, 22.2],
[3.3, 33.3],
[4.4, 44.4],
[np.nan, 55]])
# Keep only observations that are not (denoted by ~) missing
features[~np.isnan(features).any(axis=1)]
array([[ 1.1, 11.1],
[ 2.2, 22.2],
[ 3.3, 33.3],
[ 4.4, 44.4]])

# Load library
import pandas as pd
# Load data
dataframe = pd.DataFrame(features, columns=["feature_1", "feature_2"])
# Remove observations with missing values
dataframe.dropna()



# Load libraries
import numpy as np
from fancyimpute import KNN
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
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
# Predict the missing values in the feature matrix
features_knn_imputed = KNN(k=5, verbose=0).complete(standardized_features)
# Compare true and imputed values
print("True Value:", true_value)
print("Imputed Value:", features_knn_imputed[0,0])
True Value: 0.8730186114
Imputed Value: 1.09553327131



# Load library
from sklearn.preprocessing import Imputer
# Create imputer
mean_imputer = Imputer(strategy="mean", axis=0)
# Impute values
features_mean_imputed = mean_imputer.fit_transform(features)
# Compare true and imputed values
print("True Value:", true_value)
print("Imputed Value:", features_mean_imputed[0,0])
True Value: 0.8730186114
Imputed Value: -3.05837272461