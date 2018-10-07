import collections
import itertools
import locale
from urllib import parse

import spacy
from pandas.io.json import json_normalize

from src.data.url_utils import (DBPEDIA_RESOURCE_URL, get_filename_from_url,
                                quote_url)

PHYSICISTS_IMPUTE_KEYS = ['academicAdvisor', 'almaMater', 'award', 'birthPlace',
                          'categories', 'child', 'citizenship', 'deathPlace',
                          'doctoralAdvisor', 'doctoralStudent', 'field', 'influenced',
                          'influencedBy', 'knownFor', 'nationality', 'notableStudent',
                          'parent', 'residence', 'spouse', 'theorized', 'workplaces']
"""list of `str`: Physicists impute keys.

List of keys in the physicists dictionary to use for imputation of redirects.
"""

PLACES_IMPUTE_KEYS = ['categories', 'city', 'country', 'type']
"""list of `str`: Places impute keys.

List of keys in the physicists dictionary to use for imputation of redirects.
"""

# The locale needs to be set for all categories to the user’s default
# setting (typically specified in the LANG environment variable)
# to enable correct sorting of words with accents.
locale.setlocale(locale.LC_ALL, '')

# Load `en_core_web_sm` which is the default English language model
# in spacy
nlp = spacy.load('en_core_web_sm')


def json_keys_to_dict(resource_url, flat_json, json_keys,
                      ignore_urls=None):
    """Create a dictionary from a subset of the keys of a flat JSON dataframe.

    Args:
        resource_url (str): Resource URL.
        flat_json (pandas.DataFrame): pandas dataframe which has
            been `flattened` by a previous call to
            `pandas.io.json.json_normalize`.
        json_keys (list of `str`): List of JSON keys. The keys
            are semantic URLs.
        ignore_urls (list of `str`): List of URLs to ignore in
            the values of the `flat_json`.

    Returns:
        dict: Dictionary.

        The keys are the filename (or fragment) part of the
        json_keys. The values are the corresponding values in
        the flat_json.
    """

    dict_ = {}

    # loop over the keys
    for json_key in json_keys:
        flat_json_key = resource_url + '.' + json_key
        filename = get_filename_from_url(json_key)
        dict_key = filename[filename.rfind('#') + 1:]  # take fragment
        # sanitize for later merging in _merge_influences
        if dict_key == 'influenced':
            dict_key = 'influenced_'
        if flat_json_key not in flat_json:
            continue

        # loop and get the values
        for list_ in flat_json[flat_json_key].values:
            val_list = []
            for val in list_:
                if not _val_is_english(val):
                    continue

                if isinstance(val['value'], (int, float)):
                    value = val['value']
                else:  # str
                    value = val['value'].lstrip('* ')

                if ignore_urls and value in ignore_urls:
                    continue

                if not _value_has_information_content(value):
                    continue

                if isinstance(value, str) and value.startswith(
                        'http://wikidata.dbpedia.org/'):
                    continue
                val_list.append(value)

            if not val_list:
                continue
            elif len(val_list) == 1 and dict_key not in dict_:
                dict_[dict_key] = val_list[0]
            elif (len(val_list) == 2 and
                  not any(isinstance(v, str) for v in val_list) and
                  dict_key not in dict_):
                # two values with float and int should take
                # take the float (some lat and long)
                val_list = [v for v in val_list if isinstance(v, float)]
                dict_[dict_key] = val_list[0]
            else:
                if not all(isinstance(v, str) for v in val_list):
                    # multiple values here must all be of 'str' type
                    # (some child and spouse have int values)
                    val_list = [v for v in val_list if
                                not isinstance(v, (int, float))]
                if dict_key not in dict_:
                    val_list.sort(key=locale.strxfrm)
                    dict_[dict_key] = '|'.join(val_list)
    return dict_


def _val_is_english(val):
    language = val.get('lang')
    if not language or language == 'en':
        return True
    return False


def _value_has_information_content(value):
    if not isinstance(value, str):
        return True
    if (not value or
            value == '*' or
            value.startswith('* \n') or
            value.endswith('\n*') or
            value.startswith('--') or
            '_family' in value or  # e.g. child contains this
            value in spacy.lang.en.STOP_WORDS):
        return False
    return True


def json_values_to_dict(resource, flat_json, json_values):
    """Create a dictionary from a subset of the values of a flat JSON dataframe.

    Args:
        resource_url (str): Resource URL.
        flat_json (pandas.DataFrame): pandas dataframe which has
            been `flattened` by a previous call to
            `pandas.io.json.json_normalize`.
        json_keys (list of `str`): List of JSON values. The values
            are semantic URLs.

    Returns:
        dict: Dictionary.

        The keys are the filename (or fragment) part of the
        json_values. The values are the corresponding keys in
        the flat_json.
    """

    dict_ = {}

    # loop over the values
    for json_value in json_values:
        # loop and get the keys
        key_list = []
        for json_key in flat_json:
            sep_position = json_key.rfind('.http')
            key, val = json_key[:sep_position], json_key[sep_position + 1:]
            # only consider keys other than the resource
            if json_value == val and not key == resource:
                dict_key = key
                key_list.append(dict_key)
                dict_val = get_filename_from_url(val).replace('_', ' ')
        if not key_list:
            continue
        elif len(key_list) == 1:
            dict_[dict_val] = key_list[0]
        else:
            key_list.sort(key=locale.strxfrm)
            dict_[dict_val] = '|'.join(key_list)
    return dict_


def json_categories_to_dict(flat_json):
    """Create a dictionary from the categories in a flat JSON dataframe.

    Args:
        flat_json (pandas.DataFrame): pandas dataframe which has
            been `flattened` by a previous call to
            `pandas.io.json.json_normalize`.

    Returns:
        dict: Dictionary.

        The key is `categories`. The values are the corresponding
        values in the flat_json.
    """

    dict_ = {}

    val_list = []
    for list_ in flat_json.values:
        for item in list_:
            for val in item:
                value = val['value']
                if (not isinstance(value, str) or not value.startswith(
                        DBPEDIA_RESOURCE_URL + 'Category:')):
                    continue
                val_list.append(value)
    if val_list:
        val_list.sort(key=locale.strxfrm)
        dict_['categories'] = '|'.join(val_list)
    return dict_


def find_resource_url(flat_json):
    """Find the resource URL in a flat JSON dataframe.

    The resource is the is the value to the left of `sameAs`
    field in the flattened pandas dataframe.

    Args:
        flat_json (pandas.DataFrame): pandas dataframe which has
            been `flattened` by a previous call to
            `pandas.io.json.json_normalize`.

    Returns:
        str: The resource URL.
    """

    # resource is the value to the left of `sameAs` field
    OWL_SAME_AS = 'http://www.w3.org/2002/07/owl#sameAs'

    for val in flat_json.columns.values:
        if OWL_SAME_AS in val:
            resource = val.split('.' + OWL_SAME_AS)[0]
            # print(resource)
            return resource
    assert(False)  # all json files should have `sameAs` field


def get_source_url(resource_url):
    """Get the source URL from the resource URL.

    Args:
        resource_url (str): The resource URL.

    Returns:
        str: The source URL.
    """

    parsed_url = parse.urlparse(resource_url)
    filename = get_filename_from_url(resource_url)
    return ('{uri.scheme}://{uri.netloc}/'.format(uri=parsed_url) +
            'data/' + filename + '.json')


def construct_resource_urls(data, keys):
    """Construct resource URLs from data.

    Args:
        data (list of `dict`): List of dicts containing data.
        keys (list of `str`): List of keys in the dictionaries
            to get the resource URLs from.

    Returns:
        list of `str`: List of URLs.
    """

    urls_to_check = set()
    for key in keys:
        for datum in data:
            text = datum.get(key)
            if not text or isinstance(text, (int, float)):
                continue

            texts = text.split('|')
            urls = (item for item in texts if item.startswith(
                    DBPEDIA_RESOURCE_URL))
            non_urls = (item for item in texts if not item.startswith('http://')
                        and not item.startswith('https://'))
            non_urls_to_urls = (DBPEDIA_RESOURCE_URL +
                                item.replace(' ', '_') for item in non_urls)
            urls = itertools.chain(urls, non_urls_to_urls)

            for url in urls:
                quoted_url = quote_url(url)
                urls_to_check.add(quoted_url)

    urls_to_check = list(urls_to_check)
    return urls_to_check


def impute_redirect_filenames(data, keys, redirect_urls):
    """Impute the filenames from redirected URLs in the data.

    Args:
        data (list of `dict`): List of dicts containing data.
        keys (list of `str`): List of keys in the dictionaries
            to use for imputation.
        redirect_urls (dict): The redirected URLs. The key is
            the original URL and the value is the redirected URL.

    Returns:
        list of `dict`: List of dicts containing imputed data.

        Identical to `data` except that it contains the filenames
        from the redirected URLs.
    """

    imputed_data = data.copy()

    for key in keys:
        for datum in data:
            text = datum.get(key)
            if not text or isinstance(text, (int, float)):
                continue

            # split up fields with these symbols
            text = text.replace(' \n* ', '|')
            text = text.replace('\n* ', '|')
            text = text.replace(' \n', '|')

            impute_texts = set()
            texts = text.split('|')
            for text in texts:
                if (not text.startswith(DBPEDIA_RESOURCE_URL) and
                        (text.startswith('http://') or text.startswith('https://'))):
                    name = text
                else:
                    if not text.startswith(DBPEDIA_RESOURCE_URL):
                        text = DBPEDIA_RESOURCE_URL + text.replace(' ', '_')
                    if text in redirect_urls:
                        name = get_filename_from_url(redirect_urls[text])
                    else:
                        name = get_filename_from_url(text)
                name = name.replace('_', ' ')

                # if not text.startswith(DBPEDIA_RESOURCE_URL):
                #    text = DBPEDIA_RESOURCE_URL + text.replace(' ', '_')
                # if text in redirect_urls:
                #    name = get_filename_from_url(redirect_urls[text])
                # else:
                #    name = get_filename_from_url(text)
                #name = name.replace('_', ' ')

                if name.startswith('Category:'):
                    name = name.replace('Category:', '')
                impute_texts.add(name)

            impute_texts = list(impute_texts)
            impute_texts.sort(key=locale.strxfrm)
            datum[key] = '|'.join(impute_texts)

    return imputed_data
