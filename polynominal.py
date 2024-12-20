import pandas
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt

path_to_file = 'summary_means.xlsx'

def train_model(X_multi_train, X_multi_test, degree = 2):
    poly_features = PolynomialFeatures(degree)

    # train
    X_train_quadratic = poly_features.fit_transform(X_multi_train)

    # test
    X_test_quadratic = poly_features.fit_transform(X_multi_test)

    # Call it and give a name
    quadratic = LinearRegression()

    # fit the model
    quadratic.fit(X_train_quadratic, y_multi_train)

    # x_single =

    # predict on train and test inputs
    y_train_predicted = quadratic.predict(X_train_quadratic)
    y_test_predicted = quadratic.predict(X_test_quadratic)

    # visualizing the difference between actual prices and predicted values
    plt.scatter(y_multi_test, y_test_predicted, color='#029386')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted issues')
    # plt.grid(axis='y', alpha=0.5)
    plt.show()

    print(f'Train evaluation for polynomial regressor with degree {degree}:\n')
    print('R^2:', metrics.r2_score(y_multi_test, y_test_predicted))
    print('MAE:', metrics.mean_absolute_error(y_multi_test, y_test_predicted))
    print('MSE:', metrics.mean_squared_error(y_multi_test, y_test_predicted))

    results_df = pandas.DataFrame({
        'Project': X_multi_test_names,
        'Actual Value': y_multi_test,
        'Predicted Value': y_test_predicted,
        'MAE': [metrics.mean_absolute_error([y_multi_test.iloc[i]], [y_test_predicted[i]]) for i in
                range(len(y_multi_test))],
        'MSE': [metrics.mean_squared_error([y_multi_test.iloc[i]], [y_test_predicted[i]]) for i in
                range(len(y_multi_test))]
    })

    file_name = f'prediction_results_degree_{degree}.xlsx'
    results_df.to_excel(file_name, index=False)

    print(f"\nРезультати збережено в файл {file_name}")


df = pandas.read_excel(path_to_file)

# Convert comma decimal separator to period
for column in df.select_dtypes(include=['object']).columns:
    df[column] = df[column].str.replace(',', '.').astype(float) if column != 'File' else df[column]

X_multi = df.drop(['ISSUES'], axis=1)
# X_multi_names = X_multi['File']
# X_multi = X_multi.drop(['File'], axis=1)
# print(X_multi_names)
selected_metrics = ['WAC', 'DIT', 'CBO', 'MPC', 'LCOM']
selected_metrics = [f'{metric}_Mean' for metric in selected_metrics]
X_multi = X_multi[selected_metrics + ['File']]

y_multi = (df['ISSUES'])

X_multi_train, X_multi_test, y_multi_train, y_multi_test = train_test_split(X_multi,
                                                                            y_multi,
                                                                            test_size = 0.4,
                                                                            random_state=4)
X_multi_train_names = X_multi_train['File']
X_multi_test_names = X_multi_test['File']

X_multi_train = X_multi_train.drop(['File'], axis=1)
X_multi_test = X_multi_test.drop(['File'], axis=1)

for deg in range(1, 5):
    train_model(X_multi_train, X_multi_test, deg)

