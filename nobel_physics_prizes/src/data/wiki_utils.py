import time
from urllib import parse

import numpy as np
import progressbar
import requests
from bs4 import BeautifulSoup
from bs4.element import NavigableString

# Blacklist links in Wikipedia List of Physicists article
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

# Force the following redirects as DBpedia is not in sync with
# Wikipedia for these names
FORCED_REDIRECTS = {
    'Ea Ea': 'Craige Schensted',
    'Ernest Mouchez': 'Amédée Mouchez',
    'Gian Carlo Wick': 'Gian-Carlo Wick',
    'Hans Adolf Buchdahl': 'Hans Adolph Buchdahl',
    'Hans Ziegler (physicist)': 'Hans Ziegler',
    'James Jeans': 'James Hopwood Jeans',
    'Kenneth Young (physicist)': 'Kenneth Young',
    'Lawrence Bragg': 'William Lawrence Bragg',
    'Raúl Rabadán': 'Raúl Rabadan',
    "Shin'ichirō Tomonaga": "Sin'ichirō Tomonaga",
    'Thales of Miletus': 'Thales',
    'William Fuller Brown Jr.': 'William Fuller Brown, Jr.',
    'Yakov Alpert': 'Yakov Lvovich Alpert',
    'Yang Chen-Ning': 'Chen-Ning Yang'
}

# Section titles in Wikipedia List of Physicists article
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

# Wikipedia English URL
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


def get_redirected_titles(titles, forced_redirects=None,
                          remove_duplicates=False, progress_bar=None):
    """Get a list of redirected links from a list of Wikipedia articles.
    Args:
        titles (list(str)): A list of titles to fetch.
        forced_redirects (dict): A mapping of titles to force. The key is
            the retrieved title from Wikipedia and the value is the known
            title in DBpedia.
        remove_duplicates (bool): If True, remove duplicates from the list.
            False, otherwise
        progress_bar (progressbar.ProgressBar): Progress bar.

    Returns:
        list(str): List of redirected article titles.

    """

    redirected_titles = []

    for i in range(len(titles)):
        redirected = False

        if isinstance(titles[i], str):
            # fetch the page
            url = WIKI_URL + titles[i].replace(' ', '_')
            time.sleep(1)  # delay to crawl responsibly
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
            except requests.exceptions.RequestException as err:
                print(err)

            # parse javascript for redirects
            if response.status_code == requests.codes.ok:
                REDIRECT = '"wgInternalRedirectTargetUrl":'
                soup = BeautifulSoup(response.text, 'lxml')
                for script_tag in soup.find_all(name='script'):
                    script_code = script_tag.string
                    if (isinstance(script_code, NavigableString) and
                            REDIRECT in script_code):
                        start = script_code.find(REDIRECT)
                        end = script_code.find('"', start + len(REDIRECT) + 1)
                        redirected_title = (
                            script_code[start + len(REDIRECT) + 1:end]
                            .replace('/wiki/', '').replace('_', ' ')
                        )
                        redirected = True

        # Some physicist names contain unicode characters which have been quoted
        # when in a url. Unquote for sorting.
        if redirected:
            redirected_titles.append(parse.unquote(redirected_title))
        elif isinstance(titles[i], str):
            redirected_titles.append(parse.unquote(titles[i]))
        else:
            redirected_titles.append(titles[i])

        if progress_bar:
            progress_bar.update(i)

    # remove duplicates
    if remove_duplicates:
        redirected_titles = list(set(redirected_titles))

    # force redirects
    for key, value in forced_redirects.items():
        if key in redirected_titles:
            redirected_titles[redirected_titles.index(key)] = value

    return redirected_titles
