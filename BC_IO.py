"""
Copyright (c) 2021, Martin Bienert
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import pandas as pd
from pyspectra.readers.read_spc import spc


class WriteDFToFile:  # Class to write pandas DataFrames into a chi/dat file
    def __init__(self, df, filename, head="", sep="  ", include_head=True):
        self.df = df.transpose()
        self.filename = filename
        self.sep = sep
        self.head = head if include_head else ''

        self.nr_rows = len(df)
        self.nr_columns = len(df.columns)

        # Thread(self.df_to_string()).start()
        self.df_to_string()

    def df_to_string(self):
        output_list = []
        for column in self.df.columns:
            output_list.append(list(self.df[column]))

        # joins all columns in rows
        output_list = map(lambda x: self.sep.join(map(lambda y: "%2.7e" % y, x)), output_list)
        body = "\n".join(output_list)

        with open(self.filename, "w+") as writefile:
            if self.head is None:
                head = ""
            else:
                head = self.head
            writefile.write(head + "\n" + body)


class ReadRamanToDF:  # Class to read Raman file into pandas DataFrames
    def __init__(self, filename, include_head=True):
        self.filename = filename
        self.include_head = include_head

        if __name__ == '__main__':  # only shows debug output when this program is directly executed
            print(f"[ReadRamanToDF] Reading {self.filename}")

        with open(self.filename) as readfile:  # reads out content of the specified file
            self.file_content = readfile.read()

        if __name__ == '__main__':  # only shows debug output when this program is directly executed
            print(f"[ReadRamanToDF] Reading {self.filename} successfully finished")

    def file_content_to_df(self):
        lines_raw = self.file_content.split("\n")  # splits up lines and removes the header
        if self.include_head:
            lines = list(map(lambda x: x.split(), lines_raw[1:-1]))  # splits up the columns of each line
        else:
            lines = list(map(lambda x: x.split(), lines_raw[:-1]))  # splits up the columns of each line

        num_columns = len(lines[0])
        try:
            column_names = ['RamanShift (cm-1)', *lines_raw[0].split()[2:]]
        except IndexError:
            column_names = ['RamanShift (cm-1)', *list('I_' + str(x) for x in range(num_columns - 1))]

        content_df = pd.DataFrame(columns=('RamanShift (cm-1)', *column_names[1:]))

        x_column = list(map(lambda x: float(x[0]), lines))
        content_df['RamanShift (cm-1)'] = x_column

        # enumerates through all columns and writes them into the content_df dataFrame
        for column_index, column_name in enumerate(column_names):
            if column_name == 'RamanShift (cm-1)':
                continue
            y_column = list(map(lambda x: float(x[column_index]), lines))

            content_df[column_name] = y_column

        return content_df, column_names  # returns the df and head (column titles in this case)


class ReadChiToDF:  # class for reading chi file into a pandas DataFrame
    def __init__(self, filename, i_column_name, include_head=True):
        self.filename = filename
        self.i_column_name = i_column_name
        self.include_head = include_head

        if __name__ == '__main__':  # only show debug output if this file is directly executed
            print(f"[ReadChiToDF] Reading {self.filename}")

        with open(filename) as file:  # open specified file and read out content
            self.file_content = file.read()

        if __name__ == '__main__':  # only show debug output if this file is directly executed
            print(f"[ReadChiToDF] Reading {self.filename} successfully finished")

    def file_content_to_df(self):
        lines_raw = self.file_content.split("\n")  # splits up lines and removes the header
        if self.include_head:
            lines = list(map(lambda x: x.split(), lines_raw[4:-1]))  # splits up the columns of each line
        else:
            lines = list(map(lambda x: x.split(), lines_raw[:-1]))  # splits up the columns of each line

        x_column = list(map(lambda x: float(x[0]), lines))  # creates list out of first column (angles)
        y_column = list(map(lambda x: float(x[1]), lines))  # creates list out of second column (intensity)

        content_df = pd.DataFrame(columns=('q', self.i_column_name))  # creates empty dataframe to store x and y column
        content_df['q'] = x_column  # writes x_column list into 'q' column
        content_df[self.i_column_name] = y_column  # writes y_column list into 'I' (or whatever was specified) column

        if __name__ == '__main__':  # only show debug output if this file is directly executed
            from pprint import pprint
            pprint(content_df.head())

        return content_df, lines_raw[:4]  # returns the dataframe and the first 4 lines (head)


def read_raman_to_df(filename: str, header=True):
    return ReadRamanToDF(filename, include_head=header).file_content_to_df()


def read_chi_to_df(filename: str, header=True, column_name="I"):
    return ReadChiToDF(filename, include_head=header, i_column_name=column_name).file_content_to_df()


def read_spc_to_df(filename: str):
    # This function was taken from the pyspectra library to be modified to this programs needs
    out = pd.DataFrame()

    f = spc.File(filename)  # Read file
    if f.dat_fmt.endswith('-xy'):
        for s in f.sub:
            x = s.x
            y = s.y

            out["RamanShift (cm-1)"] = x
            out[str(round(s.subtime))] = y
    else:
        for s in f.sub:
            x = f.x
            y = s.y

            out["RamanShift (cm-1)"] = x
            out[str(round(s.subtime))] = y

    return out, "\t".join(map(str, out.columns))
