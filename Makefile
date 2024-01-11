env:
	poetry install

preprocess:
	poetry run python3 pipeline.py preprocess --config ./config/config.yaml

exploration:
	poetry run python3 pipeline.py exploration --config ./config/config.yaml

train:
	poetry run python3 pipeline.py train --config ./config/config.yaml

validator:
	poetry run mlflow ui

test:
	poetry run python3 pipeline.py validate --config ./config/config.yaml

inference:
	poetry run python3 pipeline.py inference --config ./config/config.yaml

.PHONY:
	make env
	make preprocess
	make exploration
	make train
	make inference
	make validate
