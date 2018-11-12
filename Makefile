PREFIX = $(PWD)/local
BINDIR = $(PREFIX)/bin
SHAREDIR = $(PREFIX)/share/ocrd_typegroups_classifier
TESTDIR = $(PREFIX)/test

TOOLS = $(shell ocrd ocrd-tool ocrd-tool.json list-tools)

help:
	@echo ""
	@echo "  Targets"
	@echo ""
	@echo "    install      Install"
	@echo "    deps         Install dependencies"
	@echo "    uninstall    Remove all installed files, and subfolder in share/"

deps:
	pip3 --no-cache-dir install -r files/requirements.txt

install:
	@mkdir -p $(SHAREDIR) $(BINDIR) $(TESTDIR)
	# TODO: get files/ from github
	cp files/ocrd_typegroups_classifier.py \
	   files/var_conv2d.py \
	   files/vraec.py \
	   files/network-epoch-99-settings-011.pth \
	   files/ocrd-tool.json \
	   $(SHAREDIR)
	cp files/ocrd_typegroups_classifier $(BINDIR)
	chmod +x $(BINDIR)/ocrd_typegroups_classifier
	mkdir -p ./test
	cp files/test.sh test/
	sed -i 's,^SHAREDIR=.*,SHAREDIR="$(SHAREDIR)",' $(BINDIR)/ocrd_typegroups_classifier test/test.sh
	sed -i 's,^BINDIR=.*,BINDIR="$(BINDIR)",' $(BINDIR)/ocrd_typegroups_classifier test/test.sh

uninstall:
	rm -r --verbose --force \
	   $(BINDIR)/ocrd_typegroups_classifier \
	   $(PREFIX)/share/ocrd_typegroups_classifier
