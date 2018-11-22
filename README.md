# ocrd_typegroups_classifier

> Typegroups classifier for OCR

## Quick setup

If needed, create a virtual environment for Python 3 (it was tested
successfully with Python 3.7), activate it, and install ocrd.

```sh
virtualenv -p python3 ocrd-venv3
source ocrd-venv3/bin/activate
pip3 install ocrd
```

Install the module and its dependencies

```
make install
make deps
```

Finally, run the test:

```
sh test/test.sh
```

## Models

Models bundled with the tool:

  * network-epoch-99-settings-011.pth: TODO describe

## Performance

The smaller the `stride`, the longer the runtime, the more accurate the result. TODO


