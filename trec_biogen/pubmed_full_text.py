from asyncio import FIRST_COMPLETED, Task, create_task, wait
from asyncio.exceptions import TimeoutError
from io import BytesIO
from pathlib import Path
from re import UNICODE, compile as re_compile
from ssl import create_default_context
from typing import AsyncIterator, Collection, Iterable, Iterator, Mapping
from urllib.parse import urljoin, urlsplit

from aiohttp import ClientError, ClientSession, ClientTimeout, CookieJar, TCPConnector
from certifi import where
from pypdf import PdfReader


def iter_open_alex_pdf_urls(paper: dict) -> Iterator[str]:
    for x in paper["locations"]:
        if x["is_oa"] is True and x["pdf_url"]:
            yield x["pdf_url"]


async def safe_download(
    session: ClientSession,
    url: str,
    timeout: ClientTimeout | None = None,
) -> bytes | None:
    """
    Download from HTTP or return None in case of any response, client, or timeout error.
    """
    try:
        async with session.get(
            url=url,
            timeout=timeout,
        ) as response:
            response.raise_for_status()
            return await response.read()
    except ClientError:
        return None
    except TimeoutError:
        return None


def _remove_emojis(text: str) -> str:
    emoji_pattern = re_compile(
        "["
        "\U0001f600-\U0001f64f"  # Emoticons
        "\U0001f300-\U0001f5ff"  # Symbols and pictographs
        "\U0001f680-\U0001f6ff"  # Transport and map symbols
        "\U0001f1e0-\U0001f1ff"  # Flags (iOS)
        "\U00002702-\U000027b0"
        "\U000024c2-\U0001f251"
        "]+",
        flags=UNICODE,
    )

    return emoji_pattern.sub(r"", text)


def safe_extract_pdf_text(pdf: bytes) -> str | None:
    """
    Extract the full text from a PDF file.
    Reads the text for each page and concatenates the pages' texts.
    """
    pdf_start = pdf[:10].strip()
    if pdf_start.startswith(b"<html") or pdf_start.startswith(b"<!DOCTYPE"):
        return None
    with BytesIO(pdf) as stream:
        try:
            reader = PdfReader(stream)
            texts = (
                page.extract_text(extraction_mode="plain")
                for page in reader.pages
            )
            text = "\n".join(texts)
            text = _remove_emojis(text)
            return text
        except KeyboardInterrupt:
            return None
        except Exception:
            return None


async def safe_download_extract_pdf_text(
    session: ClientSession,
    url: str,
    timeout: ClientTimeout | None = None,
) -> str | None:
    """
    Download and extract the full text from a PDF file.
    """
    pdf = await safe_download(
        session=session,
        url=url,
        timeout=timeout,
    )
    if pdf is None:
        return None
    return safe_extract_pdf_text(pdf)


async def safe_download_extract_first_pdf_text(
    session: ClientSession,
    urls: Iterable[str],
    timeout: ClientTimeout | None = None,
) -> str | None:
    """
    Concurrently start HTTP downloads and return the first successfully downloaded and extracted PDF text.
    """
    pending: set[Task[str | None]] = {
        create_task(safe_download_extract_pdf_text(
            session=session,
            url=url,
            timeout=timeout,
        ))
        for url in urls
    }
    done: set[Task[str | None]] = set()
    try:
        while len(pending) > 0:
            done, pending = await wait(pending, return_when=FIRST_COMPLETED)
            for task in done:
                response = await task
                if response is not None:
                    for pending_task in pending:
                        pending_task.cancel()
                    return response
    except Exception as e:
        # Cancel tasks properly if process is killed.
        for pending_task in pending:
            pending_task.cancel()
        raise e
    return None


async def iter_full_text_pdf_urls(
    session: ClientSession,
    pubmed_ids: Collection[str],
    api_base_url: str = "https://api.openalex.org/",
    timeout: ClientTimeout | None = None,
) -> AsyncIterator[tuple[str, Iterator[str]]]:
    if len(pubmed_ids) > 100:
        raise ValueError("Cannot fetch more than 100 articles at once.")

    url = urljoin(api_base_url, "works")
    params = {
        "per-page": len(pubmed_ids),
        "select": ",".join(("ids", "locations")),
        "filter": ",".join((
            "has_pmid:true",
            "locations.is_oa:true",
            "pmid:"+"|".join(pubmed_ids),
        ))
    }

    try:
        async with session.get(
            url=url,
            params=params,
            timeout=timeout,
        ) as response:
            response.raise_for_status()
            json = await response.json()
    except ClientError:
        for pubmed_id in pubmed_ids:
            yield pubmed_id, iter([])
        return
    except TimeoutError:
        for pubmed_id in pubmed_ids:
            yield pubmed_id, iter([])
        return

    if not isinstance(json, dict):
        raise RuntimeError("Could not parse response.")

    if "results" not in json or not isinstance(json["results"], list):
        raise RuntimeError("Could not parse response.")

    pubmed_ids = set(pubmed_ids)
    for paper in json["results"]:
        pubmed_url = paper["ids"]["pmid"]
        _, _, path, _, _ = urlsplit(pubmed_url)
        pubmed_id = Path(path).name
        if pubmed_id in pubmed_ids:
            pubmed_ids.remove(pubmed_id)
            yield pubmed_id, iter_open_alex_pdf_urls(paper)
    for pubmed_id in pubmed_ids:
        yield pubmed_id, iter([])


async def iter_download_extract_full_texts(
    session: ClientSession,
    pubmed_ids: Collection[str],
    api_base_url: str = "https://api.openalex.org/",
    api_timeout: ClientTimeout | None = None,
    pdf_timeout: ClientTimeout | None = None,
) -> AsyncIterator[tuple[str, str | None]]:
    full_text_pdf_urls = iter_full_text_pdf_urls(
        session=session,
        pubmed_ids=pubmed_ids,
        api_base_url=api_base_url,
        timeout=api_timeout,
    )
    async for pubmed_id, urls in full_text_pdf_urls:
        yield pubmed_id, await safe_download_extract_first_pdf_text(
            session=session,
            urls=urls,
            timeout=pdf_timeout,
        )


async def get_full_text_dict(pubmed_ids: Collection[str]) -> Mapping[str, str | None]:
    ssl_context = create_default_context(cafile=where())
    connector = TCPConnector(ssl=ssl_context)
    cookie_jar = CookieJar(unsafe=True)
    headers = {
        "User-Agent": "Webis PubMed Full Text Indexing <heinrich.reimer@uni-jena.de>",
    }
    api_timeout = ClientTimeout(total=120, connect=15)
    pdf_timeout = ClientTimeout(total=300, connect=30)
    async with ClientSession(
        connector=connector,
        cookie_jar=cookie_jar,
        headers=headers,
    ) as session:
        return {
            pubmed_id: full_text
            async for pubmed_id, full_text in iter_download_extract_full_texts(
                session=session,
                pubmed_ids=pubmed_ids,
                api_timeout=api_timeout,
                pdf_timeout=pdf_timeout,
            )
        }
