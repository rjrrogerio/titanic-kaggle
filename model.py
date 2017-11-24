import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import Imputer

if __name__ == "__main__":
    train_data_filename = "train.csv"
    test_data_filename = "test.csv"

    mlp = MLPClassifier()
    my_imputer = Imputer()

    predictors = ['Pclass', 'Age', 'SibSp',
                  'Parch', 'Fare']

    train_data = pd.read_csv(train_data_filename)
    test_data = pd.read_csv(test_data_filename)

    train_X = pd.get_dummies(train_data[predictors])
    train_y = train_data.Survived

    test_X = test_data[predictors]

    # imputed inputs
    # NaN are replaced with real numbers
    imputed_train_X = my_imputer.fit_transform(train_X)
    imputed_test_X = my_imputer.transform(test_X)

    mlp.fit(imputed_train_X, train_y)

    y = mlp.predict(imputed_test_X)

    output_data = {
        'PassengerId': test_data.PassengerId,
        'Survived': y
    }

    pd.DataFrame(output_data,
                 columns=['PassengerId',
                          'Survived']).to_csv('output.csv', index=False)
