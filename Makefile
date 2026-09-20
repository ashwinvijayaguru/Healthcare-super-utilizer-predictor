.PHONY: install data train test api web build docker clean

install:
	pip install -r requirements.txt
	cd web && npm install

data:
	python -m ml.generate_cohort --n 60000

train:
	python -m ml.train

test:
	pytest tests/ -v

api:
	uvicorn api.main:app --reload --port 8000

web:
	cd web && npm run dev

build:
	cd web && npm run build

docker:
	docker compose up --build

demo: data train build
	uvicorn api.main:app --port 8000

clean:
	rm -rf data/*.parquet data/*.db artifacts/*.joblib artifacts/*.json web/dist
