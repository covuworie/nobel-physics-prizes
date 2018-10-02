import time
from concurrent.futures import as_completed
from random import random
from urllib import parse

import pandas as pd
import progressbar
import requests
from requests_futures.sessions import FuturesSession

DBPEDIA_RESOURCE_URL = 'http://dbpedia.org/resource/'
"""str: DBpedia resource URL.

The URL path that all DBpedia resources are contained in.
These redirect automatically to the page.
"""

DBPEDIA_DATA_URL = 'http://dbpedia.org/data/'
"""str: DBpedia data URL.

The URL path that all DBpedia JSON data are contained in.
These do not redirect.
"""


def get_redirect_urls(urls_to_check, url_cache_path=None, max_workers=2,
                      timeout=10, progress_bar=None):
    """Get redirect urls using a `requests` head request.

    If a `url_cache_path` is provided and a URL in the `urls_to_check` is
    found in the cache, then the request is skipped and the values from
    the cache are returned. This function also uses exponential backoff
    if a too many requests exception (429) is encountered.

    Args:
        urls_to_check (list of `str`): List of urls to request.
        url_cache_path (str, optional): Defaults to None. Path of the csv file
            where the URL cache of known mappings is located.
        max_workers (int, optional): Defaults to 2. Number of workers to
            use in the thread pool.
        timeout (int, optional): Defaults to 10. Maximum timeout to allow for
            any request.
        progress_bar (progressbar.ProgressBar, optional): Defaults to None.
            Progress bar.

    Returns:
        dict: Dictionary of URL mappings.

        The keys are the original URLs and the values are the redirected URLs.
        Both the URLs are returned quoted.

    """

    urls = urls_to_check.copy()
    redirect_urls = {}
    if url_cache_path:
        cache = pd.read_csv(url_cache_path)
        redirect_urls = dict(zip(cache.url, cache.redirect_url))
        redirect_urls_quoted = [quote_url(url) for url in redirect_urls.keys()]
        urls = list(set(urls_to_check) - set(redirect_urls_quoted))

    futures = {}
    with FuturesSession(max_workers=max_workers) as session:
        if progress_bar:
            progress_bar.start()

        for url in urls:
            future = session.head(url, timeout=timeout, allow_redirects=True)
            futures[future] = url

        num_iters = 0
        for future in as_completed(futures):
            try:
                response = future.result()
                response.raise_for_status()
                if response.status_code == requests.codes.ok:
                    url = parse.unquote(futures[future])
                    redirect_urls[url] = parse.unquote(response.url)
            except requests.exceptions.RequestException as err:
                print(err)

            num_iters += 1
            if progress_bar:
                progress_bar.update(num_iters)

        if progress_bar:
            progress_bar.finish()

    return redirect_urls


def fetch_json_data(urls_to_fetch, max_workers=2, timeout=10,
                    progress_bar=None):
    """Fetch JSON data from a list of URLs.

    Args:
        urls_to_fetch (list of `str`): List of URLs to request.
        max_workers (int, optional): Defaults to 2. Number of workers to
            use in the thread pool.
        timeout (int, optional): Defaults to 10. Maximum timeout to allow for
            any request.
        progress_bar (progressbar.ProgressBar, optional): Defaults to None.
            Progress bar.

    Returns:
        dict: A dictionary of JSON data.

        The key is the URL for the JSON data and the second value is the JSON
        dict.

    """

    futures = {}
    with FuturesSession(max_workers=max_workers) as session:
        if progress_bar:
            progress_bar.start()

        for url in urls_to_fetch:
            future = session.get(url, timeout=timeout,
                                 background_callback=_parse_json)
            futures[future] = url

        data = {}
        num_iters = 0
        for future in as_completed(futures):
            try:
                response = future.result()
                response.raise_for_status()
                data[futures[future]] = response.data
            except requests.exceptions.RequestException as err:
                print(err)

            num_iters += 1
            if progress_bar:
                progress_bar.update(num_iters)

        if progress_bar:
            progress_bar.finish()

    return data


def _parse_json(session, response):
    if response.status_code == requests.codes.ok:
        response.data = response.json()


def urls_progress_bar(num_urls_to_check, banner_text='Fetching: ', marker='█'):
    """Create a urls progress bar for tracking the fetching progress.

    Args:
        num_urls_to_check (int): The number of urls to be requested.
        banner_text (str, optional): Defaults to 'Fetching: '. Banner text
            to show.
        marker (str, optional): Defaults to '█'. Marker to show the progress.

    Returns:
        progress_bar (progressbar.ProgressBar, optional): Defaults to None. Progress bar.

    """

    widgets = [
        banner_text, progressbar.Counter(),
        ' / ' + str(num_urls_to_check) + ' urls',
        ' ', progressbar.Bar(marker=marker),
        ' ', progressbar.Percentage(),
        ' ', progressbar.Timer(),
        ' ', progressbar.ETA()
    ]

    bar = progressbar.ProgressBar(max_value=num_urls_to_check,
                                  widgets=widgets, redirect_stdout=True)
    return bar


def quote_url(url):
    """Quote a url.

    Args:
        url (str): URL.

    Returns:
        str: Quoted URL.

    """

    pathname = get_pathname_from_url(url)
    filename = get_filename_from_url(url)
    filename = parse.quote(filename)
    quoted_url = pathname + filename
    return quoted_url


def unquote_url(url):
    """Unquote a url.

    Args:
        url (str): URL.

    Returns:
        str: Unquoted URL.

    """

    pathname = get_pathname_from_url(url)
    filename = get_filename_from_url(url)
    filename = parse.unquote(filename)
    unquoted_url = pathname + filename
    return unquoted_url


def get_pathname_from_url(url):
    """Get the pathname from a URL.

    Args:
        url (str): URL.

    Returns:
        str: Pathname of the URL.
    """

    return url[:url.rfind('/') + 1:]


def get_filename_from_url(url):
    """Get the filename from a URL.

    Args:
        url (str): URL

    Returns:
        str: Filename of the URL.
    """

    return url[url.rfind('/') + 1:]
