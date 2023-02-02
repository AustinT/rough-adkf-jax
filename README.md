# Python Template

First attempt that writing a simple version of ADKF in `jax`.

## Development

**!!PLEASE READ THIS SECTION BEFORE COMMITING ANY CODE TO THIS REPO!!**

### Installation

To create the environment:

```bash
conda env create -f environment.yml
```

To update:

```bash
conda env update --file environment.yml --prune
```

### Formatting

Use pre-commit to enforce formatting, large file checks, etc.

If not already installed in your environment, run:

```bash
conda install pre-commit
```

To install the precommit hooks:

```bash
pre-commit install
```

Now a series of useful checks will be run before any commit.

### Testing

`pytest` is used to check for code correctness.
Tests can be run with the following line of code:

```bash
python -m pytest tests/
```

**!!Before commiting code or merging to the main branch, please run the line of code above!!**
