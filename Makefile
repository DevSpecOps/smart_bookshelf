PYTHON ?= python
JUPYTER ?= jupyter

.PHONY: all data eda eval notebook clean help

help:
	@echo "Targets:"
	@echo "  data      download dataset into archive/"
	@echo "  eda       generate EDA plots"
	@echo "  eval      train models and compute offline metrics"
	@echo "  notebook  build and execute the report notebook"
	@echo "  all       run every step sequentially"
	@echo "  clean     remove generated figures and metric tables"

all: data eda eval notebook

data:
	$(PYTHON) scripts/download_data.py

eda:
	$(PYTHON) scripts/run_eda.py

eval:
	$(PYTHON) src/evaluate.py

notebook:
	$(PYTHON) scripts/build_notebook.py
	$(JUPYTER) nbconvert --to notebook --execute --inplace notebooks/report.ipynb

clean:
	$(PYTHON) -c "from pathlib import Path; [p.unlink() for p in Path('outputs/figures').glob('*.png')]"
	$(PYTHON) -c "from pathlib import Path; [p.unlink() for p in Path('outputs/metrics').glob('*.csv')]"