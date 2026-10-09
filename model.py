import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

def createModel():
    training_df = pd.read_csv("training_data.csv")
    game_ids = training_df["GAME_ID"]
    X = training_df.drop(labels=["GAME_ID", "label"], axis="columns")
    y = training_df["label"]

    gary = xgb.XGBRegressor(n_estimators=300,
                             max_depth=1,
                             learning_rate=0.05,
                             random_state=1
                             )

    x_train, x_test, y_train, y_test, game_ids_train, game_ids_test = train_test_split(
        X,
        y,
        game_ids,
        test_size=0.20,
        random_state=1
    )

    gary.fit(X=x_train, y=y_train)

    predictions = gary.predict(X=x_test)
    # mae = mean_absolute_error(y_test, predictions)
    # rmse = np.sqrt(mean_squared_error(y_test, predictions))

    predictions_df = pd.DataFrame({
        "GAME_ID": game_ids_test.values,
        "prediction": predictions,
        "actual": y_test
        })
    predictions_df.to_csv("predictions.csv", header=True, mode="w")
    # print(f"MAE is {mae} and RMSE is {rmse}")

    # naive_prediction = y_train.mean()
    # baseline_predictions = np.full_like(y_test, naive_prediction)
    # baseline_mae = mean_absolute_error(y_test, baseline_predictions)
    # print(f"Baseline MAE: {baseline_mae}")

createModel()