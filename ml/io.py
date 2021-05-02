""" Package for the import and export of data

ImportData:
    __init__
    import_text
"""

import os

class ImportData:
    def __init__(self):
        return

    def import_text(self, directory_path, n):
        """Imports a series of text files

        Given a path toa directory, which should only contain a series of
        plain text files, this will return a list of the contents of the first n
        files as strings.
        """
        data = []
        for fname in os.listdir(directory_path)[:n]:
            with open( os.path.join(directory_path, fname), encoding = 'utf-8') as f:
                data.append(f.readlines()[0])
        return data
