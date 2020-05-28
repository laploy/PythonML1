# Load library
import pandas as pd

# Create URL
url = 'https://raw.githubusercontent.com/laploy/PythonML1/master/titanic20.csv'

# Load data as a dataframe
dataframe = pd.read_csv(url)

# Show first 5 rows
print(dataframe.head(5))

"""
   ID     Name     Sex  Age
0   1   Harris    male   22
1   2  Bradley  female   38
2   3    Laina  female   26
3   4    Heath  female   35
4   5    Henry    male   35
"""


# Load library
import pandas as pd

# Create DataFrame
dataframe = pd.DataFrame()

# Add columns
dataframe['Name'] = ['Jacky Jackson', 'Steven Stevenson']
dataframe['Age'] = [38, 25]
dataframe['Driver'] = [True, False]

# Show DataFrame
print(dataframe)

"""
               Name  Age  Driver
0     Jacky Jackson   38    True
1  Steven Stevenson   25   False
"""


import pandas as pd

data = {'name': ['Somu', 'Kiku', 'Amol', 'Lini'],
        'physics': [68, 74, 77, 78],
        'chemistry': [84, 56, 73, 69],
        'algebra': [78, 88, 82, 87]}

# create dataframe
df_marks = pd.DataFrame(data)
print('Original DataFrame\n------------------')
print(df_marks)

new_row = {'name': 'Geo', 'physics': 87, 'chemistry': 92, 'algebra': 97}
# append row to the dataframe
df_marks = df_marks.append(new_row, ignore_index=True)

print('\n\nNew row added to DataFrame\n--------------------------')
print(df_marks)

"""
Original DataFrame
------------------
   name  physics  chemistry  algebra
0  Somu       68         84       78
1  Kiku       74         56       88
2  Amol       77         73       82
3  Lini       78         69       87


New row added to DataFrame
--------------------------
   name  physics  chemistry  algebra
0  Somu       68         84       78
1  Kiku       74         56       88
2  Amol       77         73       82
3  Lini       78         69       87
4   Geo       87         92       97
"""


# Load library
import pandas as pd

# Create URL
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'

# Load data as a dataframe
dataframe = pd.read_csv(url)

# Show first 5 rows
print(dataframe.head(5))

"""
   PassengerId  Survived  Pclass  ...     Fare Cabin  Embarked
0            1         0       3  ...   7.2500   NaN         S
1            2         1       1  ...  71.2833   C85         C
2            3         1       3  ...   7.9250   NaN         S
3            4         1       1  ...  53.1000  C123         S
4            5         0       3  ...   8.0500   NaN         S

[5 rows x 12 columns]
"""


# Load library
import pandas as pd

# Create URL
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'

# Load data as a dataframe
dataframe = pd.read_csv(url)

# Show dimensions
print(dataframe.shape)

"""
(891, 12)
"""


# Load library
import pandas as pd

# Create URL
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'

# Load data as a dataframe
dataframe = pd.read_csv(url)

# Show statistics
print(dataframe.describe())

"""
       PassengerId    Survived      Pclass  ...       SibSp       Parch        Fare
count   891.000000  891.000000  891.000000  ...  891.000000  891.000000  891.000000
mean    446.000000    0.383838    2.308642  ...    0.523008    0.381594   32.204208
std     257.353842    0.486592    0.836071  ...    1.102743    0.806057   49.693429
min       1.000000    0.000000    1.000000  ...    0.000000    0.000000    0.000000
25%     223.500000    0.000000    2.000000  ...    0.000000    0.000000    7.910400
50%     446.000000    0.000000    3.000000  ...    0.000000    0.000000   14.454200
75%     668.500000    1.000000    3.000000  ...    1.000000    0.000000   31.000000
max     891.000000    1.000000    3.000000  ...    8.000000    6.000000  512.329200

[8 rows x 7 columns]
"""



# Load library
import pandas as pd

# Create URL
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'

# Load data as a dataframe
dataframe = pd.read_csv(url)

# Select first row
print(dataframe.iloc[0])

"""
PassengerId                          1
Survived                             0
Pclass                               3
Name           Braund, Mr. Owen Harris
Sex                               male
Age                                 22
SibSp                                1
Parch                                0
Ticket                       A/5 21171
Fare                              7.25
Cabin                              NaN
Embarked                             S
Name: 0, dtype: object
"""




# Load library
import pandas as pd

# Create URL
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'

# Load data as a dataframe
dataframe = pd.read_csv(url)

# Select three rows
print(dataframe.iloc[1:4])

"""
   PassengerId  Survived  Pclass  ...     Fare Cabin  Embarked
1            2         1       1  ...  71.2833   C85         C
2            3         1       3  ...   7.9250   NaN         S
3            4         1       1  ...  53.1000  C123         S

[3 rows x 12 columns]
"""





# Load library
import pandas as pd

# Create URL
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'

# Load data as a dataframe
dataframe = pd.read_csv(url)

# Set index
dataframe = dataframe.set_index(dataframe['Name'])

# Show row
print(dataframe.loc['Braund, Mr. Owen Harris'])

"""
PassengerId                          1
Survived                             0
Pclass                               3
Name           Braund, Mr. Owen Harris
Sex                               male
Age                                 22
SibSp                                1
Parch                                0
Ticket                       A/5 21171
Fare                              7.25
Cabin                              NaN
Embarked                             S
Name: Braund, Mr. Owen Harris, dtype: object
"""




# Load library
import pandas as pd

# Create URL
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'

# Load data as a dataframe
dataframe = pd.read_csv(url)

# Show top two rows where column 'sex' is 'female'
print(dataframe[dataframe['Sex'] == 'female'].head(5))

"""
   PassengerId  Survived  Pclass  ...     Fare Cabin  Embarked
1            2         1       1  ...  71.2833   C85         C
2            3         1       3  ...   7.9250   NaN         S
3            4         1       1  ...  53.1000  C123         S
8            9         1       3  ...  11.1333   NaN         S
9           10         1       2  ...  30.0708   NaN         C

[5 rows x 12 columns]
"""



# Load library
import pandas as pd

# Create URL
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'

# Load data as a dataframe
dataframe = pd.read_csv(url)

# Filter rows
x = dataframe[(dataframe['Sex'] == 'female') & (dataframe['Age'] < 10)]
print(x[['Name', 'Age']].head(5))

#df1 = df[['a','b']]

"""
                                         Name  Age
10            Sandstrom, Miss. Marguerite Rut  4.0
24              Palsson, Miss. Torborg Danira  8.0
43   Laroche, Miss. Simonne Marie Anne Andree  3.0
58               West, Miss. Constance Mirium  5.0
119         Andersson, Miss. Ellis Anna Maria  2.0
"""






# Load library
import pandas as pd

# Create URL
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'

# Load data as a dataframe
dataframe = pd.read_csv(url)

# Replace values
x = dataframe['Sex'].replace("female", "Woman")

# Show result
print(x.head(5))
"""
0     male
1    Woman
2    Woman
3    Woman
4     male
Name: Sex, dtype: object
"""






# Load library
import pandas as pd

# Create URL
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'

# Load data as a dataframe
dataframe = pd.read_csv(url)

# Replace "female" and "male with "Woman" and "Man"
x = dataframe['Sex'].replace(["female", "male"], ["Woman", "Man"]).head(5)

# Show result
print(x.head(5))

"""
0      Man
1    Woman
2    Woman
3    Woman
4      Man
Name: Sex, dtype: object
"""





# Load library
import pandas as pd

# Create URL
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'

# Load data as a dataframe
dataframe = pd.read_csv(url)

# Replace values, show two rows
x = dataframe.replace(1, "One")

# Show result
print(x.head(5))

"""
  PassengerId Survived Pclass  ...     Fare Cabin Embarked
0         One        0      3  ...   7.2500   NaN        S
1           2      One    One  ...  71.2833   C85        C
2           3      One      3  ...   7.9250   NaN        S
3           4      One    One  ...  53.1000  C123        S
4           5        0      3  ...   8.0500   NaN        S

[5 rows x 12 columns]
"""







# Load library
import pandas as pd

# Create URL
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'

# Load data as a dataframe
dataframe = pd.read_csv(url)

# Replace values, show two rows
x = dataframe.replace(r"1st", "First", regex=True)

# Show result
print(x.head(5))

"""
  PassengerId Survived Pclass  ...     Fare Cabin Embarked
0         One        0      3  ...   7.2500   NaN        S
1           2      One    One  ...  71.2833   C85        C
2           3      One      3  ...   7.9250   NaN        S
3           4      One    One  ...  53.1000  C123        S
4           5        0      3  ...   8.0500   NaN        S

[5 rows x 12 columns]
"""




# Load library
import pandas as pd

# Create URL
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'

# Load data as a dataframe
dataframe = pd.read_csv(url)

# Rename column, show two rows
x = dataframe.rename(columns={'Pclass': 'Passenger Class'}).head(2)

# Select first row
print(x.iloc[0])

"""
PassengerId                              1
Survived                                 0
Passenger Class                          3
Name               Braund, Mr. Owen Harris
Sex                                   male
Age                                     22
SibSp                                    1
Parch                                    0
Ticket                           A/5 21171
Fare                                  7.25
Cabin                                  NaN
Embarked                                 S
Name: 0, dtype: object
"""



# Load library
import pandas as pd

# Create URL
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'

# Load data as a dataframe
dataframe = pd.read_csv(url)

# Rename columns, show two rows
x = dataframe.rename(columns={'PClass': 'Passenger Class', 'Sex': 'Gender'})

# Select first row
print(x.iloc[0])

"""
PassengerId                          1
Survived                             0
Pclass                               3
Name           Braund, Mr. Owen Harris
Gender                            male
Age                                 22
SibSp                                1
Parch                                0
Ticket                       A/5 21171
Fare                              7.25
Cabin                              NaN
Embarked                             S
Name: 0, dtype: object
"""



# Load library
import collections

# Create dictionary
column_names = collections.defaultdict(str)

# Create keys
for name in dataframe.columns: column_names[name]

# Show dictionary
column_names

    defaultdict(str,
    {'Age': '',
    'Name': '',
    'PClass': '',
    'Sex': '',
    'SexCode': '',
    'Survived': ''})




# Load library
import pandas as pd

# Create URL
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'

# Load data
dataframe = pd.read_csv(url)

# Calculate statistics
print('Maximum:', dataframe['Age'].max())
print('Minimum:', dataframe['Age'].min())
print('Mean:', dataframe['Age'].mean())
print('Sum:', dataframe['Age'].sum())
print('Count:', dataframe['Age'].count())

'''
Maximum: 80.0
Minimum: 0.42
Mean: 29.69911764705882
Sum: 21205.17
Count: 714
'''







# Load library
import pandas as pd

# Create URL
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'

# Load data
dataframe = pd.read_csv(url)

# Show counts
print(dataframe.count())

'''
PassengerId    891
Survived       891
Pclass         891
Name           891
Sex            891
Age            714
SibSp          891
Parch          891
Ticket         891
Fare           891
Cabin          204
Embarked       889
dtype: int64
'''



# Load library
import pandas as pd

# Create URL
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'

# Load data
dataframe = pd.read_csv(url)

# Select unique values
x = dataframe['Sex'].unique()
print(x)

'''
['male' 'female']
'''



# Load library
import pandas as pd

# Create URL
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'

# Load data
dataframe = pd.read_csv(url)

# Show counts
x = dataframe['Sex'].value_counts()
print(x)

'''
male      577
female    314
Name: Sex, dtype: int64
'''





# Load library
import pandas as pd

# Create URL
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'

# Load data
dataframe = pd.read_csv(url)

# Show counts
x = dataframe['Pclass'].value_counts()
print(x)

'''
3    491
1    216
2    184
Name: Pclass, dtype: int64
'''





# Load library
import pandas as pd

# Create URL
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'

# Load data
dataframe = pd.read_csv(url)

# Show number of unique values
x = dataframe['Pclass'].nunique()
print(x)
# 3



# Load library
import pandas as pd

# Create URL
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'

# Load data
dataframe = pd.read_csv(url)

# Select missing values
x = dataframe[dataframe['Age'].isnull()]

print(x.head(6))

"""
    PassengerId  Survived  Pclass  ...     Fare Cabin  Embarked
5             6         0       3  ...   8.4583   NaN         Q
17           18         1       2  ...  13.0000   NaN         S
19           20         1       3  ...   7.2250   NaN         C
26           27         0       3  ...   7.2250   NaN         C
28           29         1       3  ...   7.8792   NaN         Q
29           30         0       3  ...   7.8958   NaN         S

[6 rows x 12 columns]
"""




# Load library
import pandas as pd

# Create URL
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'

# Load data
dataframe = pd.read_csv(url)

# Attempt to replace values with NaN
x = dataframe['Sex'] = dataframe['Sex'].replace('male', NaN)

print(x.head(6))

"""
Traceback (most recent call last):
  File "C:/Users/loy2020/PycharmProjects/untitled2/test.py", line 12, in <module>
    x = dataframe['Sex'] = dataframe['Sex'].replace('male', NaN)
NameError: name 'NaN' is not defined
"""




# Load library
import pandas as pd
import numpy as np

# Create URL
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'

# Load data
dataframe = pd.read_csv(url)

# Replace values with NaN
x = dataframe['Sex'] = dataframe['Sex'].replace('male', np.nan)

print(x.head(6))

"""
0       NaN
1    female
2    female
3    female
4       NaN
5       NaN
Name: Sex, dtype: object
"""




# Load library
import pandas as pd
import numpy as np

# Create URL
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'

# Load data
dataframe = pd.read_csv(url)

# Load data, set missing values
x = dataframe = pd.read_csv(url, na_values=[np.nan, 'NONE', -999])

print(x.head(6))

"""
   PassengerId  Survived  Pclass  ...     Fare Cabin  Embarked
0            1         0       3  ...   7.2500   NaN         S
1            2         1       1  ...  71.2833   C85         C
2            3         1       3  ...   7.9250   NaN         S
3            4         1       1  ...  53.1000  C123         S
4            5         0       3  ...   8.0500   NaN         S
5            6         0       3  ...   8.4583   NaN         Q

[6 rows x 12 columns]
"""





# Load library
import pandas as pd

# Create URL
url = 'https://tinyurl.com/yc5qu6wh'

# Load data
dataframe = pd.read_csv(url)

# Delete column
x = dataframe.drop('Age', axis=1)

# Show result
print('Before\n--------------------')
print(dataframe.head(2))
print('After\n--------------------')
print(x.head(2))

'''
Before
--------------------
   ID     Name     Sex  Age
0   1   Harris    male   22
1   2  Bradley  female   38
After
--------------------
   ID     Name     Sex
0   1   Harris    male
1   2  Bradley  female
'''





# Load library
import pandas as pd

# Create URL
url = 'https://tinyurl.com/yc5qu6wh'

# Load data
dataframe = pd.read_csv(url)

# Drop columns
x = dataframe.drop(['Age', 'Sex'], axis=1)

# Show result
print('Before\n--------------------')
print(dataframe.head(2))
print('After\n--------------------')
print(x.head(2))

'''
Before
--------------------
   ID     Name     Sex  Age
0   1   Harris    male   22
1   2  Bradley  female   38
After
--------------------
   ID     Name
0   1   Harris
1   2  Bradley

'''




# Load library
import pandas as pd

# Create URL
url = 'https://tinyurl.com/yc5qu6wh'

# Load data
dataframe = pd.read_csv(url)

# Drop columns
x = dataframe.drop(dataframe.columns[1], axis=1)

# Show result
print('Before\n--------------------')
print(dataframe.head(2))
print('After\n--------------------')
print(x.head(2))

'''
Before
--------------------
   ID     Name     Sex  Age
0   1   Harris    male   22
1   2  Bradley  female   38
After
--------------------
   ID     Sex  Age
0   1    male   22
1   2  female   38
'''





# Load library
import pandas as pd

# Create URL
url = 'https://tinyurl.com/yc5qu6wh'

# Load data
dataframe = pd.read_csv(url)

# Drop columns
x = dataframe
dataframe.drop(dataframe.columns[1], axis=1, inplace=True)

# Show result
print('Before\n--------------------')
print(dataframe.head(2))
print('After\n--------------------')
print(x.head(2))

'''
Before
--------------------
   ID     Sex  Age
0   1    male   22
1   2  female   38
After
--------------------
   ID     Sex  Age
0   1    male   22
1   2  female   38
'''






# Load library
import pandas as pd

# Create URL
url = 'https://tinyurl.com/yc5qu6wh'

# Load data
dataframe = pd.read_csv(url)

# Delete rows
x = dataframe[dataframe['Sex'] != 'male']

# Show result
print('Before\n--------------------')
print(dataframe.head(4))
print('After\n--------------------')
print(x.head(4))

'''
Before
--------------------
   ID     Name     Sex  Age
0   1   Harris    male   22
1   2  Bradley  female   38
2   3    Laina  female   26
3   4    Heath  female   35
After
--------------------
   ID     Name     Sex  Age
1   2  Bradley  female   38
2   3    Laina  female   26
3   4    Heath  female   35
8   9    Oscar  female   27
'''




# Load library
import pandas as pd

# Create URL
url = 'https://tinyurl.com/yc5qu6wh'

# Load data
dataframe = pd.read_csv(url)

# Delete rows
x = dataframe[dataframe['Name'] != 'Harris']

# Show result
print('Before\n--------------------')
print(dataframe.head(4))
print('After\n--------------------')
print(x.head(4))

'''
Before
Before
--------------------
   ID     Name     Sex  Age
0   1   Harris    male   22
1   2  Bradley  female   38
2   3    Laina  female   26
3   4    Heath  female   35
After
--------------------
   ID     Name     Sex  Age
1   2  Bradley  female   38
2   3    Laina  female   26
3   4    Heath  female   35
4   5    Henry    male   35

'''



# Load library
import pandas as pd

# Create URL
url = 'https://tinyurl.com/yc5qu6wh'

# Load data
dataframe = pd.read_csv(url)

# Delete rows
x = dataframe[dataframe.index != 1]

# Show result
print('Before\n--------------------')
print(dataframe.head(4))
print('After\n--------------------')
print(x.head(4))

'''
Before
--------------------
   ID     Name     Sex  Age
0   1   Harris    male   22
1   2  Bradley  female   38
2   3    Laina  female   26
3   4    Heath  female   35
After
--------------------
   ID    Name     Sex  Age
0   1  Harris    male   22
2   3   Laina  female   26
3   4   Heath  female   35
4   5   Henry    male   35

'''





# Load library
import pandas as pd

# Create URL
url = 'https://tinyurl.com/yc8fzycu'

# Load data
dataframe = pd.read_csv(url)

# Delete rows
x = dataframe.drop_duplicates()

# Show result
print('Before\n--------------------')
print(dataframe.head(4))
print('After\n--------------------')
print(x.head(4))

'''
Before
--------------------
   ID     Name     Sex  Age
0   1   Harris    male   22
1   2  Bradley  female   38
2   1   Harris    male   22
3   3    Laina  female   26
After
--------------------
   ID     Name     Sex  Age
0   1   Harris    male   22
1   2  Bradley  female   38
3   3    Laina  female   26
4   4    Heath  female   35

'''


# Load library
import pandas as pd

# Create URL
url = 'https://tinyurl.com/yc8fzycu'

# Load data
dataframe = pd.read_csv(url)

# Delete rows
# dataframe.drop_duplicates()

# Show number of rows
print("Number Of Rows In The Original DataFrame:", len(dataframe))
print("Number Of Rows After Deduping:", len(dataframe.drop_duplicates()))

'''
Number Of Rows In The Original DataFrame: 21
Number Of Rows After Deduping: 20

'''





# Load library
import pandas as pd

# Create URL
url = 'https://tinyurl.com/yc8fzycu'

# Load data
dataframe = pd.read_csv(url)

# Delete rows
x = dataframe.drop_duplicates(subset=['Sex'])

# Show result
print('Before\n--------------------')
print(dataframe.head(4))
print('After\n--------------------')
print(x.head(4))

'''
Before
--------------------
   ID     Name     Sex  Age
0   1   Harris    male   22
1   2  Bradley  female   38
2   1   Harris    male   22
3   3    Laina  female   26
After
--------------------
   ID     Name     Sex  Age
0   1   Harris    male   22
1   2  Bradley  female   38

'''



# Load library
import pandas as pd

# Create URL
url = 'https://tinyurl.com/yc8fzycu'

# Load data
dataframe = pd.read_csv(url)

# Drop duplicates
x = dataframe.drop_duplicates(subset=['Sex'], keep='last')

# Show result
print('Before\n--------------------')
print(dataframe.head(4))
print('After\n--------------------')
print(x.head(4))

'''
Before
--------------------
   ID     Name     Sex  Age
0   1   Harris    male   22
1   2  Bradley  female   38
2   1   Harris    male   22
3   3    Laina  female   26
After
--------------------
    ID     Name     Sex  Age
18  18  Charles    male   22
20  20   Fatima  female   22

'''







# Load library
import pandas as pd

# Create URL
url = 'https://tinyurl.com/yc8fzycu'

# Load data
dataframe = pd.read_csv(url)

# Group rows by the values of the column 'Sex', calculate mean
# of each group
x = dataframe.groupby('Sex').mean()

# Show result
print(x)

'''
          ID        Age
Sex                    
female  11.0  29.454545
male     9.0  23.000000

'''




# Load library
import pandas as pd

# Create URL
url = 'https://tinyurl.com/yc8fzycu'

# Load data
dataframe = pd.read_csv(url)

# Group rows
x = dataframe.groupby('Sex')

# Show result
print(x)

'''
<pandas.core.groupby.generic.DataFrameGroupBy 
object at 0x000002565E1CBB88>

'''



# Load library
import pandas as pd

# Create URL
url = 'https://tinyurl.com/ya82jqn8'

# Load data
dataframe = pd.read_csv(url)

# Group rows, count rows
x = dataframe.groupby('Survived')['Name'].count()

# Show result
print(x)

'''
Survived
0    549
1    342
Name: Name, dtype: int64
'''



# Load library
import pandas as pd

# Create URL
url = 'https://tinyurl.com/ya82jqn8'

# Load data
dataframe = pd.read_csv(url)

# Group rows, count rows
x = dataframe.groupby(['Sex','Survived'])['Age'].mean()

# Show result
print(x)

'''
Sex     Survived
female  0           25.046875
        1           28.847716
male    0           31.618056
        1           27.276022
Name: Age, dtype: float64
'''



# Load libraries
import pandas as pd
import numpy as np

# Create date range
time_index = pd.date_range('06/06/2017', periods=100000, freq='30S')

# Create DataFrame
dataframe = pd.DataFrame(index=time_index)

# Create column of random values
dataframe['Sale_Amount'] = np.random.randint(1, 10, 100000)

# Group rows by week, calculate sum per week
x = dataframe.resample('W').sum()

print('Original data\n---------------------')
print(dataframe.head(4))
print('Group rows by week, calculate sum per week\n---------------------')
print(x)

'''
Original data
---------------------
                     Sale_Amount
2017-06-06 00:00:00            8
2017-06-06 00:00:30            9
2017-06-06 00:01:00            9
2017-06-06 00:01:30            2
Group rows by week, calculate sum per week
---------------------
            Sale_Amount
2017-06-11        86460
2017-06-18       100896
2017-06-25       100996
2017-07-02       100629
2017-07-09       100186
2017-07-16        10264
'''









# Load libraries
import pandas as pd
import numpy as np

# Create date range
time_index = pd.date_range('06/06/2017', periods=100000, freq='30S')

# Create DataFrame
dataframe = pd.DataFrame(index=time_index)

# Create column of random values
dataframe['Sale_Amount'] = np.random.randint(1, 10, 100000)

# Group rows by week, calculate sum per week
x = dataframe.resample('2W').mean()

print('Original data\n---------------------')
print(dataframe.head(4))
print('\nGroup rows by 2 weeks\n---------------------')
print(x)

'''
Original data
---------------------
                     Sale_Amount
2017-06-06 00:00:00            5
2017-06-06 00:00:30            8
2017-06-06 00:01:00            5
2017-06-06 00:01:30            6

Group rows by 2 weeks
---------------------
            Sale_Amount
2017-06-11     4.984606
2017-06-25     4.968130
2017-07-09     5.005903
2017-07-23     5.047115
'''





# Load libraries
import pandas as pd
import numpy as np

# Create date range
time_index = pd.date_range('06/06/2017', periods=100000, freq='30S')

# Create DataFrame
dataframe = pd.DataFrame(index=time_index)

# Create column of random values
dataframe['Sale_Amount'] = np.random.randint(1, 10, 100000)

# Group rows by week, calculate sum per week
x = dataframe.resample('M').mean()

print('Original data\n---------------------')
print(dataframe.head(4))
print('\nGroup rows by Month\n---------------------')
print(x)

'''
Original data
---------------------
                     Sale_Amount
2017-06-06 00:00:00            5
2017-06-06 00:00:30            4
2017-06-06 00:01:00            5
2017-06-06 00:01:30            5

Group rows by Month
---------------------
            Sale_Amount
2017-06-30     4.992292
2017-07-31     4.984500
'''



# Load libraries
import pandas as pd
import numpy as np

# Create date range
time_index = pd.date_range('06/06/2017', periods=100000, freq='30S')

# Create DataFrame
dataframe = pd.DataFrame(index=time_index)

# Create column of random values
dataframe['Sale_Amount'] = np.random.randint(1, 10, 100000)

# Group rows by week, calculate sum per week
x = dataframe.resample('M', label='left').count()

print('Original data\n---------------------')
print(dataframe.head(4))
print('\nGroup rows\n---------------------')
print(x)

'''
Original data
---------------------
                     Sale_Amount
2017-06-06 00:00:00            4
2017-06-06 00:00:30            7
2017-06-06 00:01:00            9
2017-06-06 00:01:30            8

Group rows
---------------------
            Sale_Amount
2017-05-31        72000
2017-06-30        28000
'''




# Load library
import pandas as pd

# Create URL
url = 'https://tinyurl.com/yc8fzycu'

# Load data
dataframe = pd.read_csv(url)

# Print first six names uppercased
for name in dataframe['Name'][0:6]:
    print(name.upper())

'''
HARRIS
BRADLEY
HARRIS
LAINA
HEATH
HENRY
'''



# Load library
import pandas as pd

# Create URL
url = 'https://tinyurl.com/yc8fzycu'

# Load data
dataframe = pd.read_csv(url)

# Print first six names uppercased
print([name.upper() for name in dataframe['Name'][0:6]])

'''
['HARRIS', 'BRADLEY', 'HARRIS', 'LAINA', 'HEATH', 'HENRY']
'''






# Load library
import pandas as pd

# Create URL
url = 'https://tinyurl.com/ya82jqn8'

# Load data
dataframe = pd.read_csv(url)

# Create function
def uppercase(x):
    return x.upper()

# Apply function, show six rows
x = dataframe['Name'].apply(uppercase)[0:6]

# Show result
print(x)

'''
0                              BRAUND, MR. OWEN HARRIS
1    CUMINGS, MRS. JOHN BRADLEY (FLORENCE BRIGGS TH...
2                               HEIKKINEN, MISS. LAINA
3         FUTRELLE, MRS. JACQUES HEATH (LILY MAY PEEL)
4                             ALLEN, MR. WILLIAM HENRY
5                                     MORAN, MR. JAMES
Name: Name, dtype: object
'''





# Load library
import pandas as pd

# Create URL
url = 'https://tinyurl.com/ya82jqn8'

# Load data
dataframe = pd.read_csv(url)

# Group rows, apply function to groups
x = dataframe.groupby('Sex').apply(lambda x: x.count())

# Show result
print(x[['Name', 'Pclass', 'Age', 'Sex', 'Survived']])

'''
        Name  Pclass  Age  Sex  Survived
Sex                                     
female   314     314  261  314       314
male     577     577  453  577       577

[2 rows x 12 columns]
'''



# Load library
import pandas as pd

# Create DataFrame a
data_a = {'id': ['1', '2', '3'],
'first': ['Alex', 'Amy', 'Allen'],
'last': ['Anderson', 'Ackerman', 'Ali']}
dataframe_a = pd.DataFrame(data_a, columns = ['id', 'first', 'last'])

# Create DataFrame b
data_b = {'id': ['4', '5', '6'],
'first': ['Billy', 'Brian', 'Bran'],
'last': ['Bonder', 'Black', 'Balwner']}
dataframe_b = pd.DataFrame(data_b, columns = ['id', 'first', 'last'])

# Concatenate DataFrames by rows
dataframe_c =pd.concat([dataframe_a, dataframe_b], axis=0)

print('\nDataframe a\n---------------------')
print(dataframe_a)
print('\nDataframe b\n---------------------')
print(dataframe_b)
print('\nDataframe c\n---------------------')
print(dataframe_c)

'''
Dataframe a
---------------------
  id  first      last
0  1   Alex  Anderson
1  2    Amy  Ackerman
2  3  Allen       Ali

Dataframe b
---------------------
  id  first     last
0  4  Billy   Bonder
1  5  Brian    Black
2  6   Bran  Balwner

Dataframe c
---------------------
  id  first      last
0  1   Alex  Anderson
1  2    Amy  Ackerman
2  3  Allen       Ali
0  4  Billy    Bonder
1  5  Brian     Black
2  6   Bran   Balwner
'''




# Load library
import pandas as pd

# Create DataFrame a
data_a = {'id': ['1', '2', '3'],
'first': ['Alex', 'Amy', 'Allen'],
'last': ['Anderson', 'Ackerman', 'Ali']}
dataframe_a = pd.DataFrame(data_a, columns = ['id', 'first', 'last'])

# Create DataFrame b
data_b = {'id': ['4', '5', '6'],
'first': ['Billy', 'Brian', 'Bran'],
'last': ['Bonder', 'Black', 'Balwner']}
dataframe_b = pd.DataFrame(data_b, columns = ['id', 'first', 'last'])

# Concatenate DataFrames by rows
dataframe_c = pd.concat([dataframe_a, dataframe_b], axis=1)

print('\nDataframe a\n---------------------')
print(dataframe_a)
print('\nDataframe b\n---------------------')
print(dataframe_b)
print('\nDataframe c\n---------------------')
print(dataframe_c)

'''
Dataframe a
---------------------
  id  first      last
0  1   Alex  Anderson
1  2    Amy  Ackerman
2  3  Allen       Ali

Dataframe b
---------------------
  id  first     last
0  4  Billy   Bonder
1  5  Brian    Black
2  6   Bran  Balwner

Dataframe c
---------------------
  id  first      last id  first     last
0  1   Alex  Anderson  4  Billy   Bonder
1  2    Amy  Ackerman  5  Brian    Black
2  3  Allen       Ali  6   Bran  Balwner
'''



# Load library
import pandas as pd

# Create DataFrame a
data_a = {'id': ['1', '2', '3'],
'first': ['Alex', 'Amy', 'Allen'],
'last': ['Anderson', 'Ackerman', 'Ali']}
dataframe_a = pd.DataFrame(data_a, columns = ['id', 'first', 'last'])

# Create row
row = pd.Series([10, 'Chris', 'Chillon'], index=['id', 'first', 'last'])

# Append row
dataframe_b = dataframe_a.append(row, ignore_index=True)

print('\nDataframe a\n---------------------')
print(dataframe_a)
print('\nDataframe b\n---------------------')
print(dataframe_b)

'''
Dataframe a
---------------------
  id  first      last
0  1   Alex  Anderson
1  2    Amy  Ackerman
2  3  Allen       Ali

Dataframe b
---------------------
   id  first      last
0   1   Alex  Anderson
1   2    Amy  Ackerman
2   3  Allen       Ali
3  10  Chris   Chillon
'''





# Load library
import pandas as pd

# Create DataFrame employee
employee_data = {'employee_id': ['1', '2', '3', '4'],
'name': ['Issac Jones', 'Loy Keys', 'Alice Bees', 'Busaba Horton']}
dataframe_employees = pd.DataFrame(employee_data, columns = ['employee_id',
'name'])

# Create DataFrame sales
sales_data = {'employee_id': ['3', '4', '5', '6'],
'total_sales': [23456, 2512, 2345, 1455]}
dataframe_sales = pd.DataFrame(sales_data, columns = ['employee_id',
'total_sales'])

# Merge DataFrames
x = pd.merge(dataframe_employees, dataframe_sales, on='employee_id')


print('\nDataframe employee\n---------------------')
print(dataframe_employees)
print('\nDataframe sales\n---------------------')
print(dataframe_sales)
print('\nResult\n---------------------')
print(x)

'''
Dataframe employee
---------------------
  employee_id           name
0           1    Issac Jones
1           2       Loy Keys
2           3     Alice Bees
3           4  Busaba Horton

Dataframe sales
---------------------
  employee_id  total_sales
0           3        23456
1           4         2512
2           5         2345
3           6         1455

Result
---------------------
  employee_id           name  total_sales
0           3     Alice Bees        23456
1           4  Busaba Horton         2512
'''






# Load library
import pandas as pd

# Create DataFrame employee
employee_data = {'employee_id': ['1', '2', '3', '4'],
'name': ['Issac Jones', 'Loy Keys', 'Alice Bees', 'Busaba Horton']}
dataframe_employees = pd.DataFrame(employee_data, columns = ['employee_id',
'name'])

# Create DataFrame sales
sales_data = {'employee_id': ['3', '4', '5', '6'],
'total_sales': [23456, 2512, 2345, 1455]}
dataframe_sales = pd.DataFrame(sales_data, columns = ['employee_id',
'total_sales'])

# Merge DataFrames
x = pd.merge(dataframe_employees, dataframe_sales, on='employee_id', how='outer')

print('\nDataframe employee\n---------------------')
print(dataframe_employees)
print('\nDataframe sales\n---------------------')
print(dataframe_sales)
print('\nResult\n---------------------')
print(x)

'''
Result
---------------------
  employee_id           name  total_sales
0           1    Issac Jones          NaN
1           2       Loy Keys          NaN
2           3     Alice Bees      23456.0
3           4  Busaba Horton       2512.0
4           5            NaN       2345.0
5           6            NaN       1455.0
'''



# Load library
import pandas as pd

# Create DataFrame employee
employee_data = {'employee_id': ['1', '2', '3', '4'],
'name': ['Issac Jones', 'Loy Keys', 'Alice Bees', 'Busaba Horton']}
dataframe_employees = pd.DataFrame(employee_data, columns = ['employee_id',
'name'])

# Create DataFrame sales
sales_data = {'employee_id': ['3', '4', '5', '6'],
'total_sales': [23456, 2512, 2345, 1455]}
dataframe_sales = pd.DataFrame(sales_data, columns = ['employee_id',
'total_sales'])

# Merge DataFrames
x = pd.merge(dataframe_employees, dataframe_sales, on='employee_id', how='left')

print('\nDataframe employee\n---------------------')
print(dataframe_employees)
print('\nDataframe sales\n---------------------')
print(dataframe_sales)
print('\nResult\n---------------------')
print(x)

'''
Result
---------------------
  employee_id           name  total_sales
0           1    Issac Jones          NaN
1           2       Loy Keys          NaN
2           3     Alice Bees      23456.0
3           4  Busaba Horton       2512.0
'''




# Load library
import pandas as pd

# Create DataFrame employee
employee_data = {'employee_id': ['1', '2', '3', '4'],
'name': ['Issac Jones', 'Loy Keys', 'Alice Bees', 'Busaba Horton']}
dataframe_employees = pd.DataFrame(employee_data, columns = ['employee_id',
'name'])

# Create DataFrame sales
sales_data = {'employee_id': ['3', '4', '5', '6'],
'total_sales': [23456, 2512, 2345, 1455]}
dataframe_sales = pd.DataFrame(sales_data, columns = ['employee_id',
'total_sales'])

# Merge DataFrames
x = pd.merge(dataframe_employees, dataframe_sales, left_on='employee_id',
    right_on='employee_id')

print('\nResult\n---------------------')
print(x)

'''
Result
---------------------
  employee_id           name  total_sales
0           3     Alice Bees        23456
1           4  Busaba Horton         2512
'''
