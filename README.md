# 🏥 trec-biogen

Webis at [TREC 2024 BioGen](https://dmice.ohsu.edu/trec-biogen/index.html).

## Installation

1. Install [Python 3.11](https://python.org/downloads/).
2. Create and activate a virtual environment:

    ```shell
    python3.11 -m venv venv/
    source venv/bin/activate
    ```

3. Install project dependencies:

    ```shell
    pip install -e .
    ```

## Usage

Run the CLI with:

```shell
trec-biogen --help
```

### Index PubMed

TODO

### Fetch PubMed full texts

```shell
ray job submit --runtime-env ray-runtime-env.yml --no-wait -- python -m trec_biogen index-pubmed-full-texts --sample 0.01
```

## Development

Refer to the general [installation instructions](#installation) to set up the development environment and install the dependencies.
Then, also install the test dependencies:

```shell
pip install -e .[tests]
```

After having implemented a new feature, please check the code format, inspect common LINT errors, and run all unit tests with the following commands:

```shell
ruff check .                   # Code format and LINT
mypy .                         # Static typing
bandit -c pyproject.toml -r .  # Security
pytest .                       # Unit tests
```

## Contribute

If you have found an important feature missing from our tool, please suggest it by creating an [issue](https://github.com/janheinrichmerker/trec-biogen/issues). We also gratefully accept [pull requests](https://github.com/janheinrichmerker/trec-biogen/pulls)!

If you are unsure about anything, post an [issue](https://github.com/janheinrichmerker/trec-biogen/issues/new) or contact us:

- [heinrich.reimer@uni-jena.de](mailto:heinrich.reimer@uni-jena.de)

We are happy to help!

## License

This repository is released under the [MIT license](LICENSE).
Files in the `data/` directory are exempt from this license.
