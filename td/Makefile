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

ifndef with-correction
	BATCH += --eval "(setq org-export-exclude-tags '(\"correction\"))"
else
endif

files_org = $(wildcard *.org)
files_pdf = $(files_org:.org=.pdf)
files_ipynb = $(wildcard *.ipynb)
files_binder = $(addprefix ../binder-sandbox/, $(files_ipynb))

all: $(files_pdf)

%.pdf: %.org
	@echo "NOTICE: Exporting $< to pdf..."
	@TEXINPUTS=".:..:$$TEXINPUTS" $(BATCH) --visit "$<" --funcall python-export-classes > /dev/null 2>&1

ipynb: $(files_binder)

../binder-sandbox/%.ipynb: %.ipynb
	@echo "NOTICE: Running $<..."
ifndef with-correction
	jq 'del(.cells[] | select(.metadata[] == ["skip"] or .metadata[] == ["correction"]))' $< > $@
else
	jupyter nbconvert --execute --to notebook --inplace $<
	jq 'del(.cells[] | select(.metadata[] == ["skip"] or .metadata[] == ["reponse"]))' $< > $@
	sed -i -e 's/\(.*\)\("toc-hr-collapsed":\).*/\1\2 false,/' -e 's/\(.*\)\("toc-nb-collapsed":\).*/\1\2 false/' $@

endif

clean:
	@rm -rf	latex.d *.tex *.aux *.pdf *.fdb* *~ _minted*
