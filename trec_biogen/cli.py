from importlib.metadata import version
from typing import Any

from click import group, Context, Parameter, echo, option
from dotenv import load_dotenv, find_dotenv


def echo_version(
    context: Context,
    _parameter: Parameter,
    value: Any,
) -> None:
    if not value or context.resilient_parsing:
        return
    echo(version("trec-biogen"))
    context.exit()


@group()
@option(
    "-V",
    "--version",
    is_flag=True,
    callback=echo_version,
    expose_value=False,
    is_eager=True,
)
def cli() -> None:
    if find_dotenv():
        load_dotenv()


@cli.command()
@option(
    "--dry-run/",
    type=bool,
    default=False,
)
@option(
    "--refetch/",
    type=bool,
    default=False,
)
@option(
    "--sample",
    type=float,
)
def index_pubmed_full_texts(
    dry_run: bool = False,
    refetch: bool = False,
    sample: float | None = None,
) -> None:
    from trec_biogen.jobs.index_pubmed_full_text import index_pubmed_full_texts as _index_pubmed_full_texts
    _index_pubmed_full_texts(
        dry_run=dry_run,
        refetch=refetch,
        sample=sample,
    )
