TeXFile = ising.tex
PythonFile = code/*.py
RefFile = ref.bib

ising.pdf : $(TeXFile) $(PythonFile)
	xelatex -shell-escape ising.tex

again :
	xelatex -shell-escape ising.tex

full :
	xelatex -shell-escape ising.tex
	bibtex ising.aux
	xelatex -shell-escape ising.tex
	xelatex -shell-escape ising.tex

clean:
	rm -r *.aux *.bbl *.blg *.dvi *.bak *.out *.toc *.tol *.lof *.log *.lot *.synctex.gz _minted*
