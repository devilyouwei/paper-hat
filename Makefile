# ============================================================
# Makefile for HAT paper (NeurIPS 2025)
# ============================================================
# Usage:
#   make          — full build (pdflatex + bibtex)
#   make quick    — single pdflatex pass (no bib)
#   make clean    — remove build artifacts
#   make watch    — continuous rebuild on file changes (requires fswatch)
# ============================================================

MAIN   = main
LATEX  = pdflatex
BIBTEX = bibtex
VIEWER = open   # macOS; use xdg-open on Linux

# Default target: full build
.PHONY: all
all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex sections/*.tex references.bib math_commands.tex neurips_2026.sty
	$(LATEX) $(MAIN)
	$(BIBTEX) $(MAIN)
	$(LATEX) $(MAIN)
	$(LATEX) $(MAIN)

.PHONY: quick
quick:
	$(LATEX) $(MAIN)

.PHONY: view
view: all
	$(VIEWER) $(MAIN).pdf

.PHONY: clean
clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out \
	      $(MAIN).toc $(MAIN).lof $(MAIN).lot $(MAIN).fls $(MAIN).fdb_latexmk \
	      $(MAIN).synctex.gz $(MAIN).nav $(MAIN).snm $(MAIN).vrb

.PHONY: distclean
distclean: clean
	rm -f $(MAIN).pdf

.PHONY: watch
watch:
	fswatch -o $(MAIN).tex sections/*.tex references.bib math_commands.tex | xargs -n1 -I{} make quick

.PHONY: wordcount
wordcount:
	@texcount -inc -total $(MAIN).tex 2>/dev/null || echo "Install texcount for word counting"
