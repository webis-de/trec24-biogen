
from os import environ
from elasticsearch7 import AsyncElasticsearch, Elasticsearch


def elasticsearch_connection() -> Elasticsearch:
    elasticsearch_url: str = environ["ELASTICSEARCH_URL"]
    elasticsearch_username: str | None = environ.get("ELASTICSEARCH_USERNAME")
    elasticsearch_password: str | None = environ.get("ELASTICSEARCH_PASSWORD")

    elasticsearch_auth: tuple[str, str] | None
    if elasticsearch_username is not None and elasticsearch_password is None:
        raise ValueError("Must provide both username and password or neither.")
    elif elasticsearch_password is not None and elasticsearch_username is None:
        raise ValueError("Must provide both password and username or neither.")
    elif elasticsearch_username is not None and elasticsearch_password is not None:
        elasticsearch_auth = (elasticsearch_username, elasticsearch_password)
    else:
        elasticsearch_auth = None

    return Elasticsearch(
        hosts=elasticsearch_url,
        http_auth=elasticsearch_auth,
        request_timeout=60,
        read_timeout=60,
        max_retries=10,
    )


def async_elasticsearch_connection() -> AsyncElasticsearch:
    elasticsearch_url: str = environ["ELASTICSEARCH_URL"]
    elasticsearch_username: str | None = environ.get("ELASTICSEARCH_USERNAME")
    elasticsearch_password: str | None = environ.get("ELASTICSEARCH_PASSWORD")

    elasticsearch_auth: tuple[str, str] | None
    if elasticsearch_username is not None and elasticsearch_password is None:
        raise ValueError("Must provide both username and password or neither.")
    elif elasticsearch_password is not None and elasticsearch_username is None:
        raise ValueError("Must provide both password and username or neither.")
    elif elasticsearch_username is not None and elasticsearch_password is not None:
        elasticsearch_auth = (elasticsearch_username, elasticsearch_password)
    else:
        elasticsearch_auth = None

    return AsyncElasticsearch(
        hosts=elasticsearch_url,
        http_auth=elasticsearch_auth,
        request_timeout=60,
        read_timeout=60,
        max_retries=10,
    )
