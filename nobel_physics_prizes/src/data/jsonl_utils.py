import gzip
import jsonlines
import os
import shutil


def read_jsonl(file):
    """Read a JSON lines file into a list of JSON dicts.

    Args:
        file (file object): An existing file object to be read from.
            The file can either be a json lines file (extension `.jsonl`)
            or a gzip file (extension `.gz`). In the latter case the file
            will be unzipped before being read from.

    Returns:
        list: A list of JSON dicts.
    """

    filename, file_ext = os.path.splitext(file)

    # unzip the file
    if file_ext == '.gz':
        jsonl_file = filename
        with gzip.open(file, 'rb') as src, open(jsonl_file, 'wb') as dest:
            shutil.copyfileobj(src, dest)
    else:
        jsonl_file = file

    # read in the lines
    json_lines = []
    with jsonlines.open(jsonl_file, mode='r') as reader:
        for json_line in reader:
            json_lines.append(json_line)

    # delete file
    if file_ext == '.gz':
        os.remove(jsonl_file)

    return json_lines
