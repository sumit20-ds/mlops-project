.PHONY: install install-ui train serve ui test lint docker-build docker-run mlflow-ui drift baseline

install:
	pip install -r requirements.txt

install-ui:
	pip install -r requirements-ui.txt

ui:
	streamlit run streamlit_app.py

train:
	python -m src.train

train-tuned:
	python -m src.train --tune --register

baseline:
	python scripts/build_reference_stats.py

serve:
	uvicorn app.main:app --reload --port 8000

test:
	pytest tests/ -v

lint:
	ruff check src app tests sagemaker

docker-build:
	docker build -t mental-health-score-api .

docker-run:
	docker run --rm -p 8000:8000 mental-health-score-api

compose-up:
	docker compose up --build

mlflow-ui:
	mlflow ui --backend-store-uri file://$(PWD)/mlruns --port 5000

drift:
	curl -s http://localhost:8000/monitoring/drift | python -m json.tool
