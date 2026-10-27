# latexmk configuration: pdfLaTeX + biber (biblatex).  Run `latexmk` in this folder.
@default_files = ('main.tex');
$pdf_mode = 1;
$pdflatex = 'pdflatex -interaction=nonstopmode -file-line-error -synctex=1 %O %S';
$bibtex_use = 2;
$clean_ext = 'bbl run.xml synctex.gz';
