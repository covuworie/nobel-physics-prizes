import copy
import sys
from concurrent.futures import as_completed
from random import random
from urllib import parse

import pandas as pd
import progressbar
import requests
from bs4 import BeautifulSoup
from bs4.element import NavigableString
from requests_futures.sessions import FuturesSession

from src.data.url_utils import get_filename_from_url

"""list of `str`: Blacklist of links.

Blacklist links in Wikipedia `List of Physicists` article
"""
BLACKLIST_LINKS = [
    'Newcastle University',  # university
    # leads to the physicist at this foreign language link:
    # # https://tr.wikipedia.org/wiki/Victor_Twersky_(fizik%C3%A7i)
    'Ernst equation',  # equation
    'Matthew Sanders',  # not a physicist
    'Royal Prussia',    # region
    'Ricardo Carezani',  # not found in DBpedia (misspelt there?)
    'Twersky#Twersky'  # a group of people of this name
]

"""list of `str`: Section titles.

Section titles in Wikipedia `List of Physicists` article.
"""
SECTION_TITLES = [
    'Ancient times',
    'Middle_Ages',
    '15th–16th century',
    '16th–17th century',
    '17th–18th century',
    '18th–19th century',
    '19th century',
    '19th–20th century',
    '20th century',
    '20th–21st century'
]

"""str: Wikipedia English URL for old pages on Wikipedia.

The URL path that all old English Wikipedia articles are contained in.
Every article update in Wikipedia has an associated `oldid` so that the 
version can be reovered. This is important for reproducibility.
"""
WIKI_OLD_URL = 'https://en.wikipedia.org/w/index.php?title='


"""str: Wikipedia English URL.

The URL path that all English Wikipedia articles are contained in.
"""
WIKI_URL = 'https://en.wikipedia.org/wiki/'


def get_linked_article_titles(url, section_titles, blacklist_links=None):
    """Get a list of links from a Wikipedia article.
    Args:
        url (str): URL to fetch.
        section_titles (list(str)): A list of section titles from
            which to fetch the links from.
        blacklist_links (list(str)): A list of links (titles) to not fetch.
        progress_bar (progressbar.ProgressBar): Progress bar.

    Returns:
        list(str): List of linked article titles.

    """

    # fetch the page
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'lxml')

    # loop to find links
    article_titles = []
    for section_title in section_titles:
        span_id = section_title.replace(' ', '_')
        section = soup.find('span', id=span_id)
        ul = section.find_next('ul')
        for link in ul.find_all('a', href=True):
            if not link['href'].startswith('/wiki/'):
                # skip external and dead (redlink=1) links
                continue
            article_title = (
                link['href'].replace('/wiki/', '').replace('_', ' ')
            )
            if (not BLACKLIST_LINKS or
                    article_title not in BLACKLIST_LINKS):
                article_titles.append(article_title)
    return article_titles


def get_redirected_titles(
        titles_to_check, title_cache_path=None, max_workers=2, timeout=10,
        progress_bar=None):
    """Get a list of redirected links from a list of Wikipedia articles.

    Args:
        titles_to_check (list of `str`): List of titles to request.
        title_cache_path (str, optional): Defaults to None. Path of the csv file
            where the title cache of known mappings is located.
        max_workers (int, optional): Defaults to 2. Number of workers to
            use in the thread pool.
        timeout (int, optional): Defaults to 10. Maximum timeout to allow
            for any request.
        progress_bar (progressbar.ProgressBar, optional): Defaults to None.
            Progress bar.

        Returns:
            dict: Dictionary of redirected page titles.

            The keys are the original page titles and the values are the
            redirected titles.

        """

    titles = titles_to_check.copy()
    redirected_titles = {}
    if title_cache_path:
        cache = pd.read_csv(title_cache_path)
        redirected_titles = dict(zip(cache.name, cache.redirect_name))
    titles = (
        list({parse.unquote(name) for name in titles_to_check if isinstance(name, str)} -
        set(redirected_titles.keys()))
        )

    futures = {}
    with FuturesSession(max_workers=max_workers) as session:
        if progress_bar:
            progress_bar.start()

        for title in titles:
            if isinstance(title, str):
                # fetch the page
                url = WIKI_URL + title.replace(' ', '_')
                future = session.get(url, timeout=timeout)
                futures[future] = title

        num_iters = 0
        for future in as_completed(futures):
            try:
                response = future.result()
                response.raise_for_status()
            except requests.exceptions.RequestException as err:
                print(err)

            # parse javascript for redirects
            redirected_title = _parse_javascript(response)
            title = parse.unquote(futures[future])
            if redirected_title:
                redirected_titles[title] = parse.unquote(redirected_title)
            else:
                redirected_titles[title] = title

            num_iters += 1
            if progress_bar:
                progress_bar.update(num_iters)

        if progress_bar:
            progress_bar.finish()

    return redirected_titles


def _parse_javascript(response):
    if response.status_code == requests.codes.ok:
        REDIRECT = '"wgInternalRedirectTargetUrl":'
        soup = BeautifulSoup(response.text, 'lxml')

        for script_tag in soup.find_all(name='script'):
            script_code = script_tag.string
            if (isinstance(script_code, NavigableString) and
                    REDIRECT in script_code):
                start = script_code.find(REDIRECT)
                end = script_code.find(
                    '"', start + len(REDIRECT) + 1)
                redirected_title = (
                    script_code[start + len(REDIRECT) + 1:end]
                    .replace('/wiki/', '').replace('_', ' '))
                return redirected_title

    return None
