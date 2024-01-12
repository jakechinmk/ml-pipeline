

FROM compare

WORKDIR /app

ADD . /app

RUN apt-get update && apt-get install -y libgomp1

RUN pip install -r requirements.txt

EXPOSE 8000

CMD ["python", "Pipeline(memory=FastMemory(location=/var/folders/kk/dc9lcyy12nj95k9f5xqzl8_h0000gn/T/joblib),
         steps=[('numerical_imputer',
                 TransformerWrapper(exclude=None,
                                    include=['RevolvingUtilizationOfUnsecuredLines',
                                             'age', 'DebtRatio',
                                             'MonthlyIncome',
                                             'NumberOfTime30-59DaysPastDueNotWorse',
                                             'NumberOfTimes90DaysLate',
                                             'NumberOfTime60-89DaysPastDueNotWorse',
                                             'NumberOfOp...
                                            criterion='friedman_mse', init=None,
                                            learning_rate=0.1, loss='log_loss',
                                            max_depth=3, max_features=None,
                                            max_leaf_nodes=None,
                                            min_impurity_decrease=0.0,
                                            min_samples_leaf=1,
                                            min_samples_split=2,
                                            min_weight_fraction_leaf=0.0,
                                            n_estimators=100,
                                            n_iter_no_change=None,
                                            random_state=123, subsample=1.0,
                                            tol=0.0001, validation_fraction=0.1,
                                            verbose=0, warm_start=False))],
         verbose=False).py"]
