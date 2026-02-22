.PHONY: ui real-scan

PYTHON ?= python3
ifeq ($(wildcard .venv/bin/python),.venv/bin/python)
PYTHON := .venv/bin/python
endif

ui:
	$(PYTHON) -m app.ui

real-scan:
	./scripts/run_real_scan.sh
