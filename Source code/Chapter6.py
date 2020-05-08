# Load scikit-learn's datasets
from sklearn import datasets
# Load digits dataset
digits = datasets.load_digits()
# Create features matrix
features = digits.data
# Create target vector
target = digits.target
# View first observation
features[0]
array([ 0., 0., 5., 13., 9., 1., 0., 0., 0., 0., 13.,
15., 10., 15., 5., 0., 0., 3., 15., 2., 0., 11.,
8., 0., 0., 4., 12., 0., 0., 8., 8., 0., 0.,
5., 8., 0., 0., 9., 8., 0., 0., 4., 11., 0.,
1., 12., 7., 0., 0., 2., 14., 5., 10., 12., 0.,
0., 0., 0., 6., 13., 10., 0., 0., 0.])




# Load library
from sklearn.datasets import make_regression
# Generate features matrix, target vector, and the true coefficients
features, target, coefficients = make_regression(n_samples = 100,
n_features = 3,
n_informative = 3,
n_targets = 1,
noise = 0.0,
coef = True,
random_state = 1)
# View feature matrix and target vector
print('Feature Matrix\n', features[:3])
print('Target Vector\n', target[:3])
Feature Matrix
[[ 1.29322588 -0.61736206 -0.11044703]
[-2.793085 0.36633201 1.93752881]
[ 0.80186103 -0.18656977 0.0465673 ]]
Target Vector
[-10.37865986 25.5124503 19.67705609]



# Load library
from sklearn.datasets import make_classification
# Generate features matrix and target vector
features, target = make_classification(n_samples = 100,
n_features = 3,
n_informative = 3,
n_redundant = 0,
n_classes = 2,
weights = [.25, .75],
random_state = 1)
# View feature matrix and target vector
print('Feature Matrix\n', features[:3])
print('Target Vector\n', target[:3])
Feature Matrix
[[ 1.06354768 -1.42632219 1.02163151]
[ 0.23156977 1.49535261 0.33251578]
[ 0.15972951 0.83533515 -0.40869554]]
Target Vector
[1 0 0]


Load library
from sklearn.datasets import make_blobs
# Generate feature matrix and target vector
features, target = make_blobs(n_samples = 100,
n_features = 2,
centers = 3,
cluster_std = 0.5,
shuffle = True,
random_state = 1)
# View feature matrix and target vector
print('Feature Matrix\n', features[:3])
print('Target Vector\n', target[:3])
Feature Matrix
[[ -1.22685609 3.25572052]
[ -9.57463218 -4.38310652]
[-10.71976941 -4.20558148]]
Target Vector
[0 1 1]


# Load library
import matplotlib.pyplot as plt
# View scatterplot
plt.scatter(features[:,0], features[:,1], c=target)
plt.show()


# Load library
import pandas as pd
# Create URL
url = 'https://tinyurl.com/simulated_data'
# Load dataset
dataframe = pd.read_csv(url)
# View first two rows
dataframe.head(2)


# Load library
import pandas as pd
# Create URL
url = 'https://tinyurl.com/simulated_excel'
# Load data
dataframe = pd.read_excel(url, sheetname=0, header=1)
# View the first two rows
dataframe.head(2)



# Load library
import pandas as pd
# Create URL
url = 'https://tinyurl.com/simulated_json'
# Load data
dataframe = pd.read_json(url, orient='columns')
# View the first two rows
dataframe.head(2)


# Load libraries
import pandas as pd
from sqlalchemy import create_engine
# Create a connection to the database
database_connection = create_engine('sqlite:///sample.db')
# Load data
dataframe = pd.read_sql_query('SELECT * FROM data', database_connection)
# View first two rows
dataframe.head(2)