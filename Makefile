PYTHON ?= python
UV := $(shell command -v uv 2>/dev/null)

ifeq ($(UV),)
  PIP = $(PYTHON) -m pip
  RUN = $(PYTHON) -m
else
  PIP = uv pip
  RUN = uv run -m
endif

install:
	$(PIP) install -e .

sim:
	$(RUN) barcode_refiner.scripts.simulate_dataset --image images/qr.png --count 20 --out data/sim

eval:
	$(RUN) barcode_refiner.scripts.evaluate_refiners --config configs/eval.yaml

single:
	$(RUN) barcode_refiner.scripts.refine_image --image images/qr.png --points "10,10 200,20 210,210 20,200" --method edge --out results/single

demo:
	$(RUN) barcode_refiner.scripts.pipeline_demo --image images/qr.png --points "10,10 200,20 210,210 20,200" --method edge --out results/pipeline

# One-click evaluation on the provided dataset (absolute path in config)
eval-markup: install
	$(RUN) barcode_refiner.scripts.evaluate_on_markup_config --config configs/markup_eval.yaml

# After evaluation, collect top/bottom K overlays and grids
report-markup:
	$(RUN) barcode_refiner.scripts.report_topk --results results/markup_eval --method edge --k 12 --cols 3

# Detector + refiner evaluation on local dataset
eval-detector: install
	$(RUN) barcode_refiner.scripts.detector_eval_config --config configs/detector_eval.yaml

report-detector:
	$(RUN) barcode_refiner.scripts.report_topk --results results/detector_eval/opencv --method edge --k 12 --cols 3
