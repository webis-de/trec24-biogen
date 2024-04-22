from typing import Any

from click import group, Context, Parameter, echo, option
from dotenv import load_dotenv, find_dotenv

from trec_biogen import __version__ as app_version


def echo_version(
    context: Context,
    _parameter: Parameter,
    value: Any,
) -> None:
    if not value or context.resilient_parsing:
        return
    echo(app_version)
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

