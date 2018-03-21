import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def substrings_in_string(big_string, substrings):
    for substring in substrings:
        if big_string.find(substring) != -1:
            return substring
    return 'U'


def apply_ohe(data):
    """
    Applies one hot enconding to categorical fields.
    """
    return pd.get_dummies(data)


def add_agemclass(data):
    data['Age*Class'] = data['Age'] * data['Pclass']


def add_family_size(data):
    data['Family_Size'] = data['SibSp'] + data['Parch']


def add_fare_per_person(data):
    data['Fare_Per_Person'] = data['Fare'] / (data['Family_Size']+1)


def add_title(data):
    title_list = ['Mrs', 'Mr', 'Master', 'Miss',
                  'Major', 'Rev', 'Dr', 'Ms',
                  'Mlle', 'Col', 'Capt', 'Mme', 'Countess',
                  'Don', 'Jonkheer']

    data['Title'] = data['Name'].map(lambda x:
                                     substrings_in_string(x, title_list))

    # replacing all titles with mr, mrs, miss, master
    def replace_titles(x):
        title = x['Title']
        if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:
            return 'Mr'
        elif title in ['Countess', 'Mme']:
            return 'Mrs'
        elif title in ['Mlle', 'Ms']:
            return 'Miss'
        elif title == 'Dr':
            if x['Sex'] == 'Male':
                return 'Mr'
            else:
                return 'Mrs'
        else:
            return title

    data['Title'] = data.apply(replace_titles, axis=1)


def add_artificial_features(data):
    # add_agemclass(data)
    # add_fare_per_person(data)
    add_family_size(data)
    add_title(data)


if __name__ == "__main__":
    train_data_filename = "train.csv"
    test_data_filename = "test.csv"


    predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch",
              "Fare", "Embarked", "Family_Size", "Title"]

    #mlp = GradientBoostingClassifier()
    
    train_data = pd.read_csv(train_data_filename)
    test_data = pd.read_csv(test_data_filename)

    add_artificial_features(train_data)
    add_artificial_features(test_data)

    for df in train_data, test_data:
        df["Embarked"].fillna("S", inplace=True)
        for feature in "Age", "Fare":
            df[feature].fillna(train_data[feature].mean(), inplace=True)

    for df in train_data, test_data:
        df.drop("Ticket", axis=1, inplace=True)

    for df in train_data, test_data:
        df["Embarked"] = df["Embarked"].map(dict(zip(("S", "C", "Q"), (0, 1, 2))))
        df["Sex"] = df["Sex"].map(dict(zip(("female", "male"), (0, 1))))


    nominal_features = ["Pclass", "Sex", "Embarked", "Family_Size", "Title"]
    for df in train_data, test_data:
        for nominal in nominal_features:
            df[nominal] = df[nominal].astype(dtype="category")

    train_X = apply_ohe(train_data[predictors])
    train_y = train_data.Survived

    test_X = apply_ohe(test_data[predictors])

    # imputed inputs
    # NaN are replaced with 'real' values created by sklearn
    #my_imputer = Imputer()
    #train_X = my_imputer.fit_transform(train_X)
    #test_X = my_imputer.transform(test_X)

    #mlp.fit(train_X, train_y)

    #y = mlp.predict(test_X)


    forest = RandomForestClassifier(n_estimators=100,
                                criterion='gini',
                                max_depth= 5,
                                min_samples_split=10,
                                min_samples_leaf=5,
                                random_state=0)

    forest.fit(train_X, train_y)
    y = forest.predict(test_X)

    #print("Random Forest score: {0:.2}".format(forest.score(train_X, train_y)))

    output_data = {
        'PassengerId': test_data.PassengerId,
        'Survived': y
    }

    pd.DataFrame(output_data,
                 columns=['PassengerId',
                          'Survived']).to_csv('output.csv', index=False)
