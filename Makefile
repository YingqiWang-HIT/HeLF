.PHONY: install test synthetic smoke clean

install:
	python -m pip install -e ".[dev]"

test:
	pytest -q

synthetic:
	python scripts/make_synthetic_data.py --output data/synthetic --train 32 --val 8 --test 8

smoke: synthetic
	python scripts/train.py --config configs/smoke.yaml --stage A
	python scripts/train.py --config configs/smoke.yaml --stage B --resume checkpoints/smoke/stage_a_best.pt
	python scripts/train.py --config configs/smoke.yaml --stage C --resume checkpoints/smoke/stage_b_best.pt

clean:
	rm -rf data/synthetic checkpoints/smoke outputs/* .pytest_cache
