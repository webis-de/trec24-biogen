import asyncio
import gzip as gz
import io
import json
import ssl
from pprint import pprint

import aiohttp
import certifi
import requests
from pypdf import PdfReader
import logging

logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s',
                    level=logging.DEBUG)


def get_pdf_urls_for_paper(paper: dict):
    return [*map(
        lambda x: x.get("pdf_url"),
        filter(
            lambda x: x.get("is_oa") == True and x.get("pdf_url"),
            paper.get("locations")
        )
    )]


async def get_or_none(url, ssl_ctx, timeout=5):
    try:
        connector = aiohttp.TCPConnector(ssl=ssl_ctx)
        logging.info(f"Start task for url: {url}")
        async with aiohttp.ClientSession(connector=connector) as session:
            async with session.get(url, allow_redirects=True, timeout=timeout) as response:
                return await response.read(), response.status
    except aiohttp.ClientError as e:
        logging.error(f"Error for url {url}: {e}")
        return None
    except asyncio.TimeoutError as e:
        logging.error(f"Timeout for url {url}: {e}")
        return None


async def _a_download_pdf(pdf_urls, ssl_ctx, timeout=5):
    '''
    start the downloads from all given urls asynchronously and return the first successful download
    '''
    tasks = [get_or_none(pdf_url, ssl_ctx=ssl_ctx, timeout=timeout) for pdf_url in pdf_urls]
    responses = await asyncio.gather(*tasks)
    for response in responses:
        if response and response[1] == 200:
            return response[0]


async def _a_download_pdfs_for_urls_list(pdf_urls_list, ssl_ctx, timeout=5):
    '''
    call the function _a_download_pdf for each List in a LIST OF LISTS of URLs, each list contains the different urls
    for the same paper
    '''
    tasks = [(pdf_urls[0], asyncio.create_task(_a_download_pdf(pdf_urls[1], ssl_ctx=ssl_ctx, timeout=timeout))) for
             pdf_urls in pdf_urls_list]

    pdf_list = await asyncio.gather(*[task[1] for task in tasks])
    return [(task[0], pdf) for task, pdf in zip(tasks, pdf_list)]


def extract_text_from_pdf(pdf):
    '''
    extract text from pdf. Read the text for each page and concatenate them
    '''
    try:
        stream = io.BytesIO(pdf)
        reader = PdfReader(stream)
        texts = [*map(lambda x: x.extract_text(), reader.pages)]
        return " ".join(texts)
    except Exception as e:
        print(f"Error extracting text from pdf: {e}")
        return None


def download_pds_for_pubmed_ids(pubmed_ids: list[str], ssl_ctx: ssl.SSLContext, timeout: int = 5,
                                timeout_paper: int = 5) -> list[tuple]:
    '''
    download pdf full texts for a list of pubmed ids
    '''
    download_url = "https://api.openalex.org/works?per-page=100&select=ids,locations&filter=has_pmid:true,locations.is_oa:true,pmid:"

    pubmed_ids = "|".join(pubmed_ids)

    openalex_response = requests.get(download_url + pubmed_ids)

    pdf_urls_list = [
        (paper.get("ids").get("pmid"), get_pdf_urls_for_paper(paper)) for paper in
        openalex_response.json().get("results")
    ]

    pdfs = asyncio.run(_a_download_pdfs_for_urls_list(pdf_urls_list, ssl_ctx, timeout=timeout_paper))

    pdf_texts = [(pm_id, extract_text_from_pdf(pdf)) for (pm_id, pdf) in pdfs if pdf is not None]

    return pdf_texts



def index_pubmed_full_texts() -> None:
    raise NotImplementedError()


if __name__ == "__main__":
    
    API_URL = "https://api.openalex.org/works?per-page=200&select=ids,locations&filter=has_pmid:true,locations.is_oa:true&cursor="
    FILEPATH = "data/pdf_texts.jsonl.gz"
    TIMEOUT_PAPER = 60
    TIMEOUT_OPENALEX = 120
    LIMIT = 5
    downloaded_paper = 0

    ssl_ctx = ssl.create_default_context(cafile=certifi.where())

    cursor = "*"
    i = 0
    while cursor is not None:
        if LIMIT:
            i += 1
            if i > LIMIT:
                break

        print(cursor)
        response = requests.get(API_URL + cursor, timeout=TIMEOUT_OPENALEX)

        pdf_urls_list = [
            (paper.get("ids").get("pmid"), get_pdf_urls_for_paper(paper)) for paper in response.json().get("results")
        ]

        pdfs = asyncio.run(_a_download_pdfs_for_urls_list(pdf_urls_list, ssl_ctx, timeout=TIMEOUT_PAPER))

        pdf_texts = [(pm_id, extract_text_from_pdf(pdf)) for (pm_id, pdf) in pdfs if pdf is not None]

        downloaded_paper += len(pdf_texts)
        logging.info(f"Downloaded {downloaded_paper} pdfs")

        with gz.open(FILEPATH, "at") as f:
            for (pm_id, pdf_text) in pdf_texts:
                data = json.dumps({"pm_id": pm_id, "pdf_text": pdf_text}).encode("utf-8")
                f.write(f"{data}\n")

        cursor = response.json().get("meta").get("next_cursor")
