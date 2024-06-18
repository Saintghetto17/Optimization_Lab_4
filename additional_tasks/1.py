import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)


def optimize(func):
    def objective(trial):
        x = trial.suggest_float('x', 0, 5)
        y = trial.suggest_float('y', 0, 5)
        return func(x, y)

    study = optuna.create_study()
    study.optimize(objective, n_trials=1000)
    print(study.best_trial.values[0])
    return func(study.best_params['x'], study.best_params['y'])


funcs = [lambda x, y: x ** 2 + (x - 5) ** 2 + (2 * x - 4 * y) ** 2,
         lambda x, y: x ** 2 - x * y + 2 * x + y ** 2 - 4 * y + 3,
         lambda x, y: (1 - x) ** 2 + 100 * (y - x ** 2) ** 2
         ]

funcs_minimum = [12.5, -1, 0]

for i in range(len(funcs)):
    print(abs(funcs_minimum[i] - optimize(funcs[i])))


