import pandas as pd

# from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.neural_network import MLPClassifie
from sklearn.preprocessing import Imputer

if __name__ == "__main__":
    train_data_filename = "train.csv"
    test_data_filename = "test.csv"

    mlp = GradientBoostingClassifier()
    my_imputer = Imputer()

    predictors = ['Pclass', 'Sex', 'Age', 'SibSp',
                  'Parch', 'Fare', 'Embarked']

    train_data = pd.read_csv(train_data_filename)
    test_data = pd.read_csv(test_data_filename)

    train_X = pd.get_dummies(pd.get_dummies(train_data[predictors]))
    train_y = train_data.Survived

    test_X = pd.get_dummies(test_data[predictors])

    # imputed inputs
    # NaN are replaced with real numbers
    train_X = my_imputer.fit_transform(train_X)
    test_X = my_imputer.transform(test_X)

    mlp.fit(train_X, train_y)

    y = mlp.predict(test_X)

    output_data = {
        'PassengerId': test_data.PassengerId,
        'Survived': y
    }

    pd.DataFrame(output_data,
                 columns=['PassengerId',
                          'Survived']).to_csv('output.csv', index=False)
