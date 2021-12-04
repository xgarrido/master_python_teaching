# -*- mode: makefile; -*-
EMACS=emacs
BATCH=$(EMACS) --batch --no-init-file						\
		--eval "(require 'org)"						\
		--eval "(org-babel-do-load-languages 'org-babel-load-languages	\
			'((shell . t)))"					\
		--eval "(setq org-babel-use-quick-and-dirty-noweb-expansion t)" \
		--eval "(setq org-confirm-babel-evaluate nil)"			\
		--eval '(org-babel-load-file "../README.org")'                  \
		--eval '(org-babel-tangle-file "../README.org")'

files_org = $(filter-out slide_book.org, $(wildcard *.org))
files_pdf = $(files_org:.org=.pdf)

all: $(files_pdf)

%.pdf: %.org
	@echo "NOTICE: Exporting $< to pdf..."
	@TEXINPUTS=".:..:$$TEXINPUTS" $(BATCH) --visit "$<" --funcall python-export-slides
#> /dev/null 2>&1
# @rm -f ../*.sty


tar: $(files_pdf)
	@echo "NOTICE: Compressing..."
	@for dir in "td" "slides"; do \
		mkdir -p ../python_$(shell date +%Y)/$$dir; \
		make -C ../$$dir; \
		cp ../$$dir/*.pdf ../$$dir/*.ipynb ../python_$(shell date +%Y)/$$dir; \
	done
	@tar caf ../python_$(shell date +%Y).tar.gz ../python_$(shell date +%Y)

clean:
	@rm -rf	latex.d *.tex *.pdf *.fdb* *.vrb *.toc *.aux *.fls *.snm *.nav *.log *.out *.xdv *.pyg *~ _minted*
