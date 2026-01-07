PDF = PrismPair.pdf
QMD = PrismPair.qmd
SVG = PrismPair_geometry.svg 
# デフォルトターゲット
all: $(PDF)

$(PDF): $(QMD) $(PY) $(SVG)
	quarto render $(QMD) --to pdf

clean:
	rm -f $(PDF) PrismPair.hmml PrismPair.html PrismPair.quarto_ipynb
