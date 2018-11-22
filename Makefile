# pypi name of the package
PKG_NAME = ocrd_typegroups_classifier

# BEGIN-EVAL makefile-parser --make-help Makefile

help:
	@echo ""
	@echo "  Targets"
	@echo ""
	@echo "    deps       pip install -r requirements.txt"
	@echo "    install    pip install -e ."
	@echo "    uninstall  pip uninstall $(PKG_NAME)"
	@echo "    assets     Fetch test assets"
	@echo ""
	@echo "  Variables"
	@echo ""
	@echo "    PKG_NAME  pypi name of the package"

# END-EVAL

# pip install -r requirements.txt
deps:
	pip install -r requirements.txt

# pip install -e .
install:
	pip install -e .

# pip uninstall $(PKG_NAME)
uninstall:
	pip uninstall $(PKG_NAME)

# Fetch test assets
assets: test/assets

repo/assets:
	mkdir -p repo
	cd repo; git clone --depth 1 https://github.com/OCR-D/assets

# Run tests
test: repo/assets
	rm -rf tests/assets
	cp -r repo/assets/data tests/assets
	cd tests; bash test.sh

