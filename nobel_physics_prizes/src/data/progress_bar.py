import progressbar


def progress_bar(num_urls_to_check, banner_text_begin='Fetching: ',
                 banner_text_end=' urls', marker='█'):
    """Create a progress bar for tracking the progress of tasks.

    Args:
        num_urls_to_check (int): The number of urls to be requested.
        banner_text_begin (str, optional): Defaults to 'Fetching: '. Banner text
            to show at the beginning of the banner on the progress bar.
        banner_text_end (str, optional): Defaults to 'Fetching: '. Banner text
            to show at the end of the banner on the progress bar.
        marker (str, optional): Defaults to '█'. Marker to show the progress.

    Returns:
        progress_bar (progressbar.ProgressBar, optional): Defaults to None.

        Progress bar.
    """

    widgets = [
        banner_text_begin, progressbar.Counter(),
        ' / ' + str(num_urls_to_check) + banner_text_end,
        ' ', progressbar.Bar(marker=marker),
        ' ', progressbar.Percentage(),
        ' ', progressbar.Timer(),
        ' ', progressbar.ETA()
    ]

    bar = progressbar.ProgressBar(max_value=num_urls_to_check,
                                  widgets=widgets, redirect_stdout=True)
    return bar
