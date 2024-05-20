import asyncio
import gzip as gz
import io
import json
import ssl

import aiohttp
import certifi
import requests
from pypdf import PdfReader


def get_pdf_urls_for_paper(paper: dict):
    return [*map(
        lambda x: x.get("pdf_url"),
        filter(
            lambda x: x.get("is_oa") == True and x.get("pdf_url"),
            paper.get("locations")
        )
    )]

async def get_or_none(url, timeout=5):

    try:
        ssl_ctx = ssl.create_default_context(cafile=certifi.where())

        conn = aiohttp.TCPConnector(ssl=ssl_ctx)
        print("Start task for url: ", url)
        async with aiohttp.ClientSession(connector=conn) as session:
            async with session.get(url, allow_redirects=True, timeout=timeout) as response:
                return await response.read(), response.status
    except:
        return None


async def _a_download_pdf(pdf_urls, timeout=5):
    '''
    start the downloads from all given urls asynchronously and return the first successful download
    '''

    tasks = [get_or_none(pdf_url, timeout=5) for pdf_url in pdf_urls]
    print("Start tasks for pdf_urls: ", pdf_urls)
    responses = await asyncio.gather(*tasks)
    for response in responses:
        if response and response[1] == 200:
            return response[0]


async def _a_download_pdfs_for_urls_list(pdf_urls_list, timeout=5):

    tasks = [(pdf_urls[0], asyncio.create_task(_a_download_pdf(pdf_urls[1], timeout=timeout))) for pdf_urls in pdf_urls_list]

    pdf_list = await asyncio.gather(*[task[1] for task in tasks])
    return [(task[0], pdf) for task, pdf in zip(tasks, pdf_list)]

def extract_text_from_pdf(pdf):
    '''
    extract text from pdf. Read the text for each page and concatenate them
    '''
    stream = io.BytesIO(pdf)
    reader = PdfReader(stream)
    texts = [*map(lambda x: x.extract_text(), reader.pages)]
    return " ".join(texts)


if __name__ == "__main__":

    API_URL = "https://api.openalex.org/works?per-page=5&select=ids,locations&filter=has_pmid:true,locations.is_oa:true&cursor="
    FILEPATH = "data/pdf_texts.jsonl.gz"
    TIMEOUT = 5
    LIMIT = 5
    
    
    cursor = "*"
    i = 0
    while cursor is not None:
        if LIMIT:
            i += 1
            if i > LIMIT:
                break

        print(cursor)
        response = requests.get(API_URL+cursor)

        pdf_urls_list = [
            (paper.get("ids").get("pmid"), get_pdf_urls_for_paper(paper)) for paper in response.json().get("results")
        ]

        pdfs = asyncio.run(_a_download_pdfs_for_urls_list(pdf_urls_list, timeout=TIMEOUT))

        pdf_texts = [(pm_id, extract_text_from_pdf(pdf)) for (pm_id, pdf) in pdfs if pdf is not None]

        with gz.open(FILEPATH, "at") as f:
            for (pm_id, pdf_text) in pdf_texts:
                data = json.dumps({"pm_id": pm_id, "pdf_text": pdf_text}).encode("utf-8")
                f.write(f"{data}\n")

        cursor = response.json().get("meta").get("next_cursor")

