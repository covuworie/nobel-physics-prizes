import numpy as np


def nationality_to_alpha2_code(text, nationalities):
    """Create ISO 3166-1 alpha-2 country codes from nationalities.

    Use the nationality to find ISO 3166-1 alpha-2
    country codes. This function should only be called
    for a subset of the places dataframe where country is
    not defined and latitude or longitude is not (or 
    equivalently ISO 3166-1 alpha-2 country code is not
    defined).

    Args:
        text (str): Text containing nationalities.
        nationalities (pandas.Dataframe): Dataframe of
            nationalities data.

    Returns:
        `str` or `numpy.nan`: Pipe separated list of ISO 3166-1
            alpha-2 country codes if found, otherwise numpy.nan.
    """

    if isinstance(text, float):
        return np.nan

    # try as is
    texts_to_check = {text}

    # flatten all demonyms in nationalities dataframe and
    # try any of those that are found in text
    try:
        nationality_to_alpha2_code.demonyms
    except AttributeError:
        nationality_to_alpha2_code.demonyms = np.ravel(
            nationalities.drop('ISO 3166 Code', axis=1))
        nationality_to_alpha2_code.demonyms = [
            str(demonym) for demonym in nationality_to_alpha2_code.demonyms
            if str(demonym) != 'nan']

    for demonym in nationality_to_alpha2_code.demonyms:
        if demonym in text:
            texts_to_check.add(demonym)

    # remove Ireland or Irish for special case of Northern Ireland or
    # Northern Irish
    if 'Northern Ireland' in texts_to_check and 'Ireland' in texts_to_check:
        texts_to_check.remove('Ireland')
    if 'Northern Irish' in texts_to_check and 'Irish' in texts_to_check:
        texts_to_check.remove('Irish')

    # also try with an 's' on the end
    if text.endswith('s'):
        texts_to_check.add(text[:-1])

    alpha2_codes = set()
    for text_to_check in texts_to_check:
        # check all columns except ISO 3166 Code
        for column in nationalities.columns[1:]:
            nationality_present = nationalities[
                text_to_check == nationalities[column]]
            if not nationality_present.empty:
                alpha2_codes.add(
                    nationality_present[
                        'ISO 3166 Code'].values[0])

    if not alpha2_codes:
        return np.nan

    alpha2_codes = list(alpha2_codes)
    alpha2_codes.sort()
    alpha2_codes = '|'.join(alpha2_codes)
    return alpha2_codes
