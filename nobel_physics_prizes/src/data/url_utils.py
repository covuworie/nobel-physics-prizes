import time
from random import random
from urllib import parse

import pandas as pd
import progressbar
import requests

DBPEDIA_RESOURCE_URL = 'http://dbpedia.org/resource/'
"""str: DBpedia resource URL.

The URL path that all DBpedia are contained in. These
redirect automatically to the page.
"""


def get_redirect_urls(urls_to_check, url_cache_path=None, max_retries=3,
                      max_backoff=32, timeout=10, progress_bar=None):
    """Get redirect urls using a `requests` head request.

    If a `url_cache_path` is provided and a URL in the `urls_to_check` is
    found in the cache, then the request is skipped and the values from
    the cache are returned. This function also uses exponential backoff
    if a too many requests exception (429) is encountered.

    Args:
        urls_to_check (list of `str`): List of urls to request.
        url_cache_path (str, optional): Defaults to None. Path of the csv file
            where the URL cache of known mappings is located.
        max_retries (int, optional): Defaults to 3. Maximum number of retries
            for failed request.
        max_backoff (int, optional): Defaults to 32. Maximum backoff in seconds
            to allow.
        timeout (int, optional): Defaults to 10. Maximum timeout to allow for
            any request.
        progress_bar (progressbar.ProgressBar, optional): Defaults to None.
            Progress bar.

    Returns:
        dict: Dictionary of URL mappings.

        The keys are the original URLs and the values are the redirected URLs.
        Both the URLs are returned quoted.

    """

    redirect_urls = {}
    if url_cache_path and isinstance(url_cache_path, str):
        try:
            cache = pd.read_csv(url_cache_path)
            redirect_urls = dict(zip(cache.url, cache.redirect_url))
        except FileNotFoundError:
            pass

    with requests.Session() as session:
        if progress_bar:
            progress_bar.start()

        for i in range(len(urls_to_check)):
            url = urls_to_check[i]
            if url in redirect_urls:
                continue

            tries = 0
            while tries < max_retries:
                try:
                    response = session.head(
                        url, timeout=timeout, allow_redirects=True)
                    response.raise_for_status()
                    if response.status_code == requests.codes.ok:
                        redirect_urls[parse.unquote(
                            url)] = parse.unquote(response.url)
                        break
                except requests.exceptions.RequestException as err:
                    print(err)
                    if response.status_code == requests.codes.not_found:
                        break
                    if response.status_code == requests.codes.too_many_requests:
                        # exponential backoff to crawl responsibly
                        time.sleep(min(2**tries + random(), max_backoff))
                tries += 1

            if progress_bar:
                progress_bar.update(i)
        if progress_bar:
            progress_bar.finish()

    return redirect_urls


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
    """Quote a url if it contains non-ascii characters.

    Args:
        url (str): URL.

    Returns:
        str: Quoted URL if it contains non-ascii characters. 

        The original URL is returned if the URL only contains
        ascii characters.
    """

    pathname = get_pathname_from_url(url)
    filename = get_filename_from_url(url)
    if not all(ord(char) < 128 for char in filename):
        filename = parse.quote(filename)
    quoted_url = pathname + filename
    return quoted_url


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
