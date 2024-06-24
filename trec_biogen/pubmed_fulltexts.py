from asyncio import FIRST_COMPLETED, Task, create_task, wait
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from ssl import create_default_context
from typing import Any, AsyncIterable, AsyncIterator, Collection, Iterable, Iterator
from urllib.parse import urljoin, urlsplit

from aiohttp import ClientError, ClientResponseError, ClientSession, ClientTimeout, CookieJar, TCPConnector
from certifi import where
from elasticsearch7 import AsyncElasticsearch, Elasticsearch
from elasticsearch7.helpers import async_streaming_bulk
from elasticsearch7_dsl import Search
from elasticsearch7_dsl.query import Exists
from more_itertools import chunked
from pypdf import PdfReader
from tqdm.asyncio import tqdm

from trec_biogen.elasticsearch import async_elasticsearch_connection, elasticsearch_connection
from trec_biogen.pubmed import Article


async def safe_download(session: ClientSession, url: str) -> bytes | None:
    """
    Download from HTTP or return None in case of any response, client, or timeout error.
    """
    try:
        async with session.get(url) as response:
            response.raise_for_status()
            return await response.read()
    except ClientResponseError:
        return None
    except ClientError:
        return None
    except TimeoutError:
        return None


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
            return "\n".join(texts)
        except Exception:
            return None


async def safe_download_extract_pdf_text(session: ClientSession, url: str) -> str | None:
    """
    Download and extract the full text from a PDF file.
    """
    pdf = await safe_download(
        session=session,
        url=url,
    )
    if pdf is None:
        return None
    return safe_extract_pdf_text(pdf)


async def safe_download_extract_first_pdf_text(session: ClientSession, urls: Iterable[str]) -> str | None:
    """
    Concurrently start HTTP downloads and return the first successfully downloaded and extracted PDF text.
    """
    pending: set[Task[str | None]] = {
        create_task(safe_download_extract_pdf_text(session, url))
        for url in urls
    }
    done: set[Task[str | None]] = set()
    while len(pending) > 0:
        done, pending = await wait(pending, return_when=FIRST_COMPLETED)
        for task in done:
            response = await task
            if response is not None:
                for pending_task in pending:
                    pending_task.cancel()
                return response
    return None


async def download_extract_full_text(session: ClientSession, article: Article, urls: Iterable[str]) -> Article:
    """
    Return an article with the extracted full text, if any.
    """
    full_text = await safe_download_extract_first_pdf_text(
        session=session,
        urls=urls,
    )
    article.full_text = full_text
    article.last_fetched_full_text = datetime.now(UTC)
    return article


def iter_open_alex_pdf_urls(article: dict) -> Iterator[str]:
    for x in article["locations"]:
        if x["is_oa"] is True and x["pdf_url"]:
            yield x["pdf_url"]


async def get_full_text_pdf_urls(
    session: ClientSession,
    articles: Collection[Article],
    api_base_url: str = "https://api.openalex.org/",
) -> AsyncIterator[tuple[Article, Iterator[str]]]:
    if len(articles) > 100:
        raise ValueError("Cannot fetch more than 100 articles at once.")

    url = urljoin(api_base_url, "works")
    params = {
        "per-page": len(articles),
        "select": ",".join(("ids", "locations")),
        "filter": ",".join((
            "has_pmid:true",
            "locations.is_oa:true",
            "pmid:"+"|".join(article.pubmed_id for article in articles),
        ))
    }

    try:
        async with session.get(url=url, params=params) as response:
            response.raise_for_status()
            json = await response.json()
    except ClientResponseError:
        for article in articles:
            yield article, iter([])
        return
    except ClientError:
        for article in articles:
            yield article, iter([])
        return
    except TimeoutError:
        for article in articles:
            yield article, iter([])
        return

    if not isinstance(json, dict):
        raise RuntimeError("Could not parse response.")

    if "results" not in json or not isinstance(json["results"], list):
        raise RuntimeError("Could not parse response.")

    article_dict = {
        article.pubmed_id: article
        for article in articles
    }
    for paper in json["results"]:
        pubmed_url = paper["ids"]["pmid"]
        _, _, path, _, _ = urlsplit(pubmed_url)
        pubmed_id = Path(path).name
        if pubmed_id in article_dict:
            yield article_dict.pop(pubmed_id), iter_open_alex_pdf_urls(paper)
    for article in article_dict.values():
        yield article, iter([])


async def iter_full_text_pdf_urls(
    session: ClientSession,
    articles: Iterable[Article],
    api_base_url: str = "https://api.openalex.org/",
    chunk_size: int = 100,
) -> AsyncIterator[tuple[Article, Iterator[str]]]:
    for chunk in chunked(articles, chunk_size):
        async for article, urls in get_full_text_pdf_urls(
            session=session,
            articles=chunk,
            api_base_url=api_base_url,
        ):
            yield article, urls


async def iter_download_extract_full_texts(
    session: ClientSession,
    articles: Iterable[Article],
    api_base_url: str = "https://api.openalex.org/",
    chunk_size: int = 100,
) -> AsyncIterator[Article]:
    full_text_pdf_urls = iter_full_text_pdf_urls(
        session=session,
        articles=articles,
        api_base_url=api_base_url,
        chunk_size=chunk_size,
    )
    async for article, urls in full_text_pdf_urls:
        yield await download_extract_full_text(
            session=session,
            article=article,
            urls=urls,
        )


async def index_pubmed_full_texts(
    session: ClientSession,
    elasticsearch: Elasticsearch,
    async_elasticsearch: AsyncElasticsearch,
    dry_run: bool = False,
    refetch: bool = False,
) -> None:
    Article.init(using=elasticsearch)

    search: Search = Article.search(using=elasticsearch)
    if not refetch:
        search = search.filter(~Exists(field="last_fetched_full_text"))
    total: int = search.count()  # type: ignore

    if refetch:
        search = search.sort("last_fetched_full_text")

    search = search.extra(_source=["pubmed_id"])

    search = search.params(
        request_timeout=300,
        scroll="30m",
        preserve_order=False,
        size=1000,
    )

    articles: Iterable[Article] = search.scan()  # type: ignore

    full_text_articles: AsyncIterable[Article] = iter_download_extract_full_texts(
        session=session,
        articles=articles,
    )

    # Convert to actions that can be processed by Elasticsearch.
    actions: AsyncIterable[dict] = (
        {
            **{
                "doc" if k == "_source" else k: v
                for k, v in article.to_dict(include_meta=True).items()
            },
            # Ensure that we only update, not create new items.
            "_op_type": "update",
        }
        async for article in full_text_articles
    )
    if dry_run:
        actions = tqdm(  # type: ignore
            actions,
            total=total,
            desc="Fetching full texts",
            unit="article",
        )
        async for _ in actions:
            pass
        return

    results: AsyncIterable[tuple[bool, Any]] = async_streaming_bulk(  # noqa: F821
        client=async_elasticsearch,
        actions=actions,
        raise_on_error=True,
        raise_on_exception=True,
        max_retries=10,
        chunk_size=10,
    )
    results = tqdm(  # type: ignore
        results,
        total=total,
        desc="Fetching full texts",
        unit="article",
    )
    async for _ in results:
        pass


async def default_index_pubmed_full_texts(
    dry_run: bool = False,
    refetch: bool = False,
) -> None:
    from aiohttp import ClientSession
    from trec_biogen.pubmed_fulltexts import index_pubmed_full_texts
    ssl_context = create_default_context(cafile=where())
    connector = TCPConnector(ssl_context=ssl_context)
    cookie_jar = CookieJar(unsafe=True)
    headers = {
        "User-Agent": "Webis PubMed Full Text Indexer",
    }
    timeout = ClientTimeout(total=15, connect=3)
    async with ClientSession(
        connector=connector,
        cookie_jar=cookie_jar,
        headers=headers,
        timeout=timeout,
    ) as session:
        async with async_elasticsearch_connection() as async_elasticsearch:
            await index_pubmed_full_texts(
                session=session,
                elasticsearch=elasticsearch_connection(),
                async_elasticsearch=async_elasticsearch,
                dry_run=dry_run,
                refetch=refetch,
            )
