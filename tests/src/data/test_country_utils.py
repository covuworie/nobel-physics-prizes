import numpy as np
import pandas as pd
import pytest

from src.data.country_utils import nationality_to_alpha2_code


@pytest.fixture(scope='module')
def read_nationalities():
    nationalities = pd.read_csv(
        'nobel_physics_prizes/data/processed/Countries-List.csv',
        keep_default_na=False)
    nationalities = nationalities.replace('', np.nan)
    return nationalities


def test_nationality_to_alpha2_code_single(
        read_nationalities):
    text = 'Turkey'
    alpha2 = nationality_to_alpha2_code(
        text, read_nationalities)
    assert(alpha2 == 'TR')

    text = 'Turkish'
    alpha2 = nationality_to_alpha2_code(
        text, read_nationalities)
    assert(alpha2 == 'TR')

    text = 'Turk'
    alpha2 = nationality_to_alpha2_code(
        text, read_nationalities)
    assert(alpha2 == 'TR')

    text = 'Turks'
    alpha2 = nationality_to_alpha2_code(
        text, read_nationalities)
    assert(alpha2 == 'TR')


def test_nationality_to_alpha2_code_multiple(
        read_nationalities):
    text = 'Turkey and United States of America'
    alpha2 = nationality_to_alpha2_code(
        text, read_nationalities)
    assert(alpha2 == 'TR|US')

    text = 'Turkish American'
    alpha2 = nationality_to_alpha2_code(
        text, read_nationalities)
    assert(alpha2 == 'TR|US')

    text = 'Turk and American'
    alpha2 = nationality_to_alpha2_code(
        text, read_nationalities)
    assert(alpha2 == 'TR|US')

    text = 'Turkish Americans'
    alpha2 = nationality_to_alpha2_code(
        text, read_nationalities)
    assert(alpha2 == 'TR|US')


def test_nationality_to_alpha2_code_not_found(
        read_nationalities):
    text = '3M'
    alpha2 = nationality_to_alpha2_code(
        text, read_nationalities)
    assert(np.isnan(alpha2))


def test_nationality_to_alpha2_code_text_is_float(
        read_nationalities):
    text = np.nan
    alpha2 = nationality_to_alpha2_code(
        text, read_nationalities)
    assert(np.isnan(alpha2))


def test_nationality_to_alpha2_code_pipes_to_periods_for_tokenization(
        read_nationalities):
    text = 'Counties of Northern Ireland|County Down|'
    'The Troubles in County Down|Ulster'
    alpha2 = nationality_to_alpha2_code(
        text, read_nationalities)
    assert(alpha2 == 'GB')


def test_nationality_to_alpha2_code_northern_ireland(
        read_nationalities):
    text = 'Duke of Northern Ireland'
    alpha2 = nationality_to_alpha2_code(
        text, read_nationalities)
    assert(alpha2 == 'GB')

    text = 'Ireland'
    alpha2 = nationality_to_alpha2_code(
        text, read_nationalities)
    assert(alpha2 == 'IE')

    text = 'Irish'
    alpha2 = nationality_to_alpha2_code(
        text, read_nationalities)
    assert(alpha2 == 'IE')

    text = 'Northern Irish'
    alpha2 = nationality_to_alpha2_code(
        text, read_nationalities)
    assert(alpha2 == 'GB')

    text = 'Northern Ireland and The Republic of Ireland'
    alpha2 = nationality_to_alpha2_code(
        text, read_nationalities)
    assert(alpha2 == 'GB|IE')

    text = 'Northern Ireland and the Irish'
    alpha2 = nationality_to_alpha2_code(
        text, read_nationalities)
    assert(alpha2 == 'GB|IE')

    # The following tests should really return
    # both 'GB' and 'IE'. However, I have forced
    # the code to just return 'GB' for performance
    # reasons. I can easily resolve the issue by
    # using named entity recognition, but it is very
    # slow. Anyhow, it is very rare to find a physicist
    # who has both of these nationalities.
    text = 'Northern Ireland and Ireland'
    alpha2 = nationality_to_alpha2_code(
        text, read_nationalities)
    assert(alpha2 == 'GB')

    text = 'Northern Irish and Irish'
    alpha2 = nationality_to_alpha2_code(
        text, read_nationalities)
    assert(alpha2 == 'GB')
