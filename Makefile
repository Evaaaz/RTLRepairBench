PYTHON ?= python3
OUT ?=
PAPER_OUT ?= build/paper
PAPER_ASSET_OUT ?= $(PAPER_OUT)/assets
PAPER_FIGURES ?= fig_repairbench_pipeline fig_oracle_tiers_poster

.PHONY: help doctor check test test-full reproduce reproduce-strict smoke verify paper-assets paper export

help:
	@echo "Standalone release commands:"
	@echo "  make doctor            report required inputs and optional external tools"
	@echo "  make check             validate allowlist, hashes, paths, and credentials"
	@echo "  make test              run public-compatible scientific and release tests"
	@echo "  make test-full         run all tests, including private historical bindings"
	@echo "  make reproduce         rebuild and verify public analyses without network"
	@echo "  make reproduce-strict  require the frozen CPython 3.9.6 byte-exact result"
	@echo "  make smoke             export into a fresh temp tree and test it there"
	@echo "  make verify            run check, test, reproduce, and smoke"
	@echo "  make paper-assets      rebuild the standalone figures (and PNG previews when supported)"
	@echo "  make paper             rebuild assets, then build and QA the 4-page poster paper"
	@echo "  make export OUT=/path  create a new verified release directory"

doctor:
	$(PYTHON) tools/reproduce.py doctor

check:
	$(PYTHON) tools/export_standalone.py --check

test:
	PYTHONPATH="code:code/benchmarks" $(PYTHON) tools/run_public_tests.py

test-full:
	PYTHONPATH="code:code/benchmarks" $(PYTHON) -m unittest discover -s tests -p 'test_*.py' -v

reproduce:
	$(PYTHON) tools/reproduce.py offline --output build/reproduced

reproduce-strict:
	$(PYTHON) tools/reproduce.py offline --strict --output build/reproduced-strict

smoke:
	$(PYTHON) tools/export_standalone.py --smoke

verify: check test reproduce smoke

paper-assets:
	@mkdir -p "$(PAPER_ASSET_OUT)"
	@for figure in $(PAPER_FIGURES); do \
		if command -v latexmk >/dev/null 2>&1; then \
			latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error \
				-outdir="$(abspath $(PAPER_ASSET_OUT))" "paper/$$figure.tex" || exit 2; \
		elif command -v tectonic >/dev/null 2>&1; then \
			tectonic --outdir "$(abspath $(PAPER_ASSET_OUT))" "paper/$$figure.tex" || exit 2; \
		else \
			echo "paper asset build requires latexmk or tectonic on PATH" >&2; \
			exit 2; \
		fi; \
		test -s "$(PAPER_ASSET_OUT)/$$figure.pdf" || exit 2; \
		if command -v pdftoppm >/dev/null 2>&1; then \
			pdftoppm -png -singlefile -r 144 "$(PAPER_ASSET_OUT)/$$figure.pdf" \
				"$(PAPER_ASSET_OUT)/$$figure.preview" || exit 2; \
			mv "$(PAPER_ASSET_OUT)/$$figure.preview.png" \
				"$(PAPER_ASSET_OUT)/$$figure.png" || exit 2; \
		elif command -v magick >/dev/null 2>&1; then \
			magick -density 144 "$(PAPER_ASSET_OUT)/$$figure.pdf[0]" \
				"$(PAPER_ASSET_OUT)/$$figure.png" || exit 2; \
		elif command -v sips >/dev/null 2>&1; then \
			sips -s format png --resampleWidth 2753 "$(PAPER_ASSET_OUT)/$$figure.pdf" \
				--out "$(PAPER_ASSET_OUT)/$$figure.png" >/dev/null || exit 2; \
		else \
			echo "paper asset build requires pdftoppm, magick, or sips for the tracked PNG preview" >&2; \
			exit 2; \
		fi; \
	done

paper: paper-assets
	@mkdir -p "$(PAPER_OUT)"
	@if command -v latexmk >/dev/null 2>&1; then \
		TEXINPUTS="../$(PAPER_ASSET_OUT):" latexmk -cd -pdf -interaction=nonstopmode -halt-on-error -file-line-error \
			-outdir="$(abspath $(PAPER_OUT))" paper/paper.tex; \
	elif command -v tectonic >/dev/null 2>&1; then \
		TEXINPUTS="$(PAPER_ASSET_OUT):$(PAPER_OUT):" tectonic --keep-intermediates --outdir "$(abspath $(PAPER_OUT))" paper/paper.tex; \
	else \
		echo "paper build requires latexmk or tectonic on PATH" >&2; \
		exit 2; \
	fi
	@test -s "$(PAPER_OUT)/paper.pdf"
	$(PYTHON) tools/check_page_limit.py "$(PAPER_OUT)/paper.pdf" --minimum 3 --limit 4
	@command -v pdftotext >/dev/null 2>&1 || (echo "paper QA requires pdftotext" >&2; exit 2)
	@pdftotext -layout "$(PAPER_OUT)/paper.pdf" "$(PAPER_OUT)/paper.qa.txt"
	@! grep -Eq '\?\?|\[Comment:|TODO' "$(PAPER_OUT)/paper.qa.txt" || \
		(echo "paper QA found unresolved references or inline review comments" >&2; exit 2)

export:
	@test -n "$(OUT)" || (echo "OUT is required, for example: make export OUT=/tmp/lint-vs-semantic-release" >&2; exit 2)
	$(PYTHON) tools/export_standalone.py --output "$(OUT)"
