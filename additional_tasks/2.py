import keras
import optuna
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras import Input
from tensorflow.keras.optimizers import SGD, Adam, RMSprop, Adagrad
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.src.initializers import HeNormal

optuna.logging.set_verbosity(optuna.logging.WARNING)

data = pd.read_csv("housing.csv")
data.dropna(inplace=True)
data = data.drop(columns=["ocean_proximity"])
X = data.drop(['median_house_value'], axis=1)
Y = data['median_house_value']
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.01)


def task(optimizer, epoch, batch):
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1],)))
    model.add(Dense(batch * 2, activation='relu', kernel_initializer=HeNormal()))
    model.add(Dense(batch * 2, activation='relu', kernel_initializer=HeNormal()))
    model.add(Dense(1, kernel_initializer=HeNormal()))
    model.compile(loss=keras.losses.MeanSquaredError(), optimizer=optimizer)
    model.fit(X_train, Y_train, epochs=epoch, batch_size=batch, verbose=0)
    return model.evaluate(X_test, Y_test, batch_size=batch, verbose=0)


optimizers_configuration = {
    "Adam": {
        "learning_rate": lambda trial: trial.suggest_float("learning_rate", 0.00001, 0.01),
        "beta_1": lambda trial: trial.suggest_float("beta_1", 0.5, 0.9),
        "beta_2": lambda trial: trial.suggest_float("beta_2", 0.9, 0.999),
        "amsgrad": lambda trial: trial.suggest_int("amsgrad", 0, 1) == 1},
    "RMSprop": {
        "learning_rate": lambda trial: trial.suggest_float("learning_rate", 0.00001, 0.01),
        "rho": lambda trial: trial.suggest_float("rho", 0.1, 0.9),
        "centered": lambda trial: trial.suggest_int("centered", 0, 1) == 1},
    "Adagrad": {
        "learning_rate": lambda trial: trial.suggest_float("learning_rate", 0.00001, 0.01),
        "initial_accumulator_value": lambda trial: trial.suggest_float("initial_accumulator_value", 0.05, 0.25),
    }
}


def loss_calculus(optimizer_raw, params):
    optimizer_backed = optimizer_raw(**params)
    return task(optimizer_backed, 64, 256)


def objective_Adam(trial):
    params = {key: func(trial) for key, func in optimizers_configuration["Adam"].items()}
    return loss_calculus(Adam, params)


def objective_RMSprop(trial):
    params = {key: func(trial) for key, func in optimizers_configuration["RMSprop"].items()}
    return loss_calculus(RMSprop, params)


def objective_Adagrad(trial):
    params = {key: func(trial) for key, func in optimizers_configuration["Adagrad"].items()}
    return loss_calculus(Adagrad, params)


study_Adam = optuna.create_study()
study_Adam.optimize(objective_Adam, n_trials=100)

study_RMSprop = optuna.create_study()
study_RMSprop.optimize(objective_RMSprop, n_trials=100)

study_Adagrad = optuna.create_study()
study_Adagrad.optimize(objective_Adagrad, n_trials=100)

best_params_Adam = {key: study_Adam.best_params[key] for key in optimizers_configuration["Adam"].keys()}
print("Adam")
print("Best_loss: ", study_Adam.best_trial.values[0])
print("Params: ", best_params_Adam)

best_params_RMSprop = {key: study_RMSprop.best_params[key] for key in optimizers_configuration["RMSprop"].keys()}
print("RMSprop")
print("Best_loss: ", study_RMSprop.best_trial.values[0])
print("Params: ", best_params_RMSprop)

best_params_Adagrad = {key: study_Adagrad.best_params[key] for key in optimizers_configuration["Adagrad"].keys()}
print("Adagrad")
print("Best_loss: ", study_Adagrad.best_trial.values[0])
print("Params: ", best_params_Adagrad)
