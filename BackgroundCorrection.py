"""
Copyright (c) 2021, Martin Bienert
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""


############## BEGIN OF IMPORTS ##############

from tkinter.filedialog import askopenfilenames, askopenfilename
from pyspectra.readers.read_spc import spc
from tkinter import Tk
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from scipy.linalg import cholesky
from scipy import sparse
from os import getcwd
import pandas as pd
import scipy as sp
import numpy as np
import time

from pprint import pprint

############### END OF IMPORTS ###############


############## BEGIN OF OPTIONS ##############

test_row_index = ''
plot_data = True

dat_file_separator = '\t'
include_header = True

jar_correction = False
jar_scaling_range = (0, -1)  # Set range as (start_value, end_value), e.g. (0, 100)

wave_min = 180
wave_max = 3395

baseline_itermax = 100
baseline_lambda = 1E5
baseline_ratio = 0.01

readfile_ui_file_types = [("spc files", "*.spc"),
                          ("chi files", "*.chi"),
                          ("txt raman files", "*.txt"),
                          ("dat files", "*.dat"),
                          ("raman files", "*.raman")]


############### END OF OPTIONS ###############


def timeit(function):  # decorator function to record execution time of a function
    def timed(*args, **kwargs):
        start_time = time.time()
        return_value = function(*args, **kwargs)
        end_time = time.time()

        print(f'[TIMEIT: {function.__name__}]: {round(end_time - start_time, 2)} seconds')

        return return_value

    return timed


def sanitize_test_row_index():
    global test_row_index

    if test_row_index == "":
        return

    try:
        test_row_index = int(test_row_index)
    except ValueError:
        raise ValueError("Test row index could not be converted to int. Make sure to use a valid column index.")


############### BEGIN OF IO CLASSES ###############

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


################ END OF IO CLASSES ################


class Algorithms:
    @staticmethod
    def arpls(y, lam=baseline_lambda, ratio=baseline_ratio, itermax=baseline_itermax) -> np.ndarray:
        r"""
        Baseline correction using asymmetrically
        reweighted penalized least squares smoothing
        Sung-June Baek, Aaron Park, Young-Jin Ahna and Jaebum Choo,
        Analyst, 2015, 140, 250 (2015)

        Abstract

        Baseline correction methods based on penalized least squares are successfully
        applied to various spectral analyses. The methods change the weights iteratively
        by estimating a baseline. If a signal is below a previously fitted baseline,
        large weight is given. On the other hand, no weight or small weight is given
        when a signal is above a fitted baseline as it could be assumed to be a part
        of the peak. As noise is distributed above the baseline as well as below the
        baseline, however, it is desirable to give the same or similar weights in
        either case. For the purpose, we propose a new weighting scheme based on the
        generalized logistic function. The proposed method estimates the noise level
        iteratively and adjusts the weights correspondingly. According to the
        experimental results with simulated spectra and measured Raman spectra, the
        proposed method outperforms the existing methods for baseline correction and
        peak height estimation.

        :param y: input data (i.e. chromatogram of spectrum)
        :param lam: parameter that can be adjusted by user. The larger lambda is,
                    the smoother the resulting background, z
        :param ratio: wheighting deviations: 0 < ratio < 1, smaller values allow less negative values
        :param itermax: number of iterations to perform
        :return: the fitted background vector

        """
        assert itermax > 0, f"itermax parameter must be greater than 0, but is {itermax}"

        N = len(y)
        D = sp.sparse.eye(N, format='csc')
        D = D[1:] - D[:-1]  # numpy.diff( ,2) does not work with sparse matrix. This is a workaround.
        D = D[1:] - D[:-1]

        H = lam * D.T * D
        w = np.ones(N)
        for i in range(itermax):
            W = sp.sparse.diags(w, 0, shape=(N, N))
            WH = sp.sparse.csc_matrix(W + H)
            cholesky_matrix = cholesky(WH.todense())
            C = sparse.csc_matrix(cholesky_matrix)
            fsolve = sparse.linalg.spsolve(C.T, w * y)
            z = sparse.linalg.spsolve(C, fsolve)
            d = y - z
            dn = d[d < 0]
            m = np.mean(dn)
            s = np.std(dn)
            wt = 1. / (1 + np.exp(2 * (d - (2 * s - m)) / s))
            if np.linalg.norm(w - wt) / np.linalg.norm(w) < ratio:
                break
            w = wt
        return z

    @staticmethod
    def als(y, lam=baseline_lambda, ratio=baseline_ratio, itermax=baseline_itermax) -> np.ndarray:
        r"""
        edit
        Implements an Asymmetric Least Squares Smoothing
        baseline correction algorithm (P. Eilers, H. Boelens 2005)

        Baseline Correction with Asymmetric Least Squares Smoothing
        based on https://github.com/vicngtor/BaySpecPlots

        Baseline Correction with Asymmetric Least Squares Smoothing
        Paul H. C. Eilers and Hans F.M. Boelens
        October 21, 2005

        Description from the original documentation:

        Most baseline problems in instrumental methods are characterized by a smooth
        baseline and a superimposed signal that carries the analytical information: a series
        of peaks that are either all positive or all negative. We combine a smoother
        with asymmetric weighting of deviations from the (smooth) trend get an effective
        baseline estimator. It is easy to use, fast and keeps the analytical peak signal intact.
        No prior information about peak shapes or baseline (polynomial) is needed
        by the method. The performance is illustrated by simulation and applications to
        real data.

        :param y: input data (i.e. chromatogram of spectrum)
        :param lam: parameter that can be adjusted by user. The larger lambda is,
                    the smoother the resulting background, z
        :param ratio: wheighting deviations: 0 < ratio < 1, smaller values allow less negative values
        :param itermax: number of iterations to perform
        :return: the fitted background vector

        """
        L = len(y)
        # D = sparse.csc_matrix(np.diff(np.eye(L), 2))
        D = sparse.eye(L, format='csc')
        D = D[1:] - D[:-1]  # numpy.diff( ,2) does not work with sparse matrix. This is a workaround.
        D = D[1:] - D[:-1]
        D = D.T
        w = np.ones(L)
        for i in range(itermax):
            W = sparse.diags(w, 0, shape=(L, L))
            Z = W + lam * D.dot(D.T)
            z = spsolve(Z, w * y)
            w = ratio * (y > z) + (1 - ratio) * (y < z)
        return z


class BackgroundCorrection:
    readfile_options = {'initialdir': getcwd(),
                        'filetypes': readfile_ui_file_types}

    def __init__(self):
        sanitize_test_row_index()

        data_frames, files, heads = self.read_files()
        heads = self.extend_headers(heads)

        for df, filename, head in zip(data_frames, files, heads):
            self.process_data(df, filename, head)

    def read_files(self):
        data_frames = []
        heads = []

        main_window = Tk()  # opens tkinter window to let user select files to process
        files = askopenfilenames(**self.readfile_options, title='Select files to process')
        main_window.destroy()  # close window after selection is done

        print("[ReadFiles] Files selected:")
        print("\n".join(files))

        for file in files:
            file_content, file_head = self.read_file(file)
            data_frames.append(file_content)
            heads.append(file_head)

        return data_frames, files, heads  # returns both the list of dataFrames and the file list

    def read_file(self, file):
        file_ext = file.split('.')[1]

        if file_ext in ['chi', 'dat']:
            file_content, file_head = read_chi_to_df(file)
        elif file_ext in ['txt', 'raman']:
            file_content, file_head = read_raman_to_df(file)
            if test_row_index == '':
                file_head = ["", "", "", dat_file_separator.join(file_head)]
            else:
                file_head = ["", "", "", dat_file_separator.join([file_head[0], file_head[test_row_index]])]
        elif file_ext == "spc":
            file_content, file_head = read_spc_to_df(file)
            file_head = ["", "", "", file_head]
        else:
            NotImplementedError(f"The used file type has not yet been implemented!")

        return file_content, file_head

    def read_jar_reference(self):
        file = askopenfilename(**self.readfile_options, title='Select file with reference data')

        file_ext = file.split('.')[1]

        if file_ext in ['chi', 'dat']:
            file_content, _ = read_chi_to_df(file)
        elif file_ext in ['txt', 'raman']:
            file_content, _ = read_raman_to_df(file)
        elif file_ext == "spc":
            file_content, _ = read_spc_to_df(file)
        else:
            NotImplementedError(f"The used file type has not yet been implemented!")

        return file_content

    def extend_headers(self, headers):
        for index, head in enumerate(headers):
            header_extention_line2 = "PXRD_raman_background_correction_V2.3.py"
            header_extention_line3 = f"Lambda = {baseline_lambda}, Ratio = {baseline_ratio}, " \
                                     f"Itermax = {baseline_itermax}"

            head[1] += header_extention_line2
            head[2] += header_extention_line3
            headers[index] = "\n".join(head)

        return headers

    def add_baseline_diff(self, df: pd.DataFrame, column_name: str, return_df: pd.DataFrame, x_column_name: str = None):
        """
        Function for extending the return_df dataFrame with the processed values from waxs_df for a certain probe

        :param df: dataFrame with density data of the probes
        :param column_name: name of the probe (e.g. 'I_243')
        :param x_column_name: name of x column
        :param return_df: dataFrame that will later be written into the output file
        :return: the return_df with updated values AND baseline
        """

        column = df[column_name]
        baseline = Algorithms.arpls(column.to_numpy())

        intensity_corrected = np.array(column - baseline)

        if x_column_name is not None:
            # Norm intensities to area under intensity curve
            intensity_corrected_area = abs(np.trapz(y=intensity_corrected, x=df[x_column_name].to_numpy()))
            intensity_corrected_normed = intensity_corrected / intensity_corrected_area
            print(f"Final scale area: {intensity_corrected_area}")
            pprint(f"Final scale result:\n{intensity_corrected_normed}")

            return_df[column_name] = intensity_corrected_normed  # add difference to return df
        else:
            return_df[column_name] = intensity_corrected

        return return_df, baseline, intensity_corrected  # return edited return_df and the baseline

    def apply_wave_range(self, df: pd.DataFrame, column_name: str, min_selection=None, max_selection=None):
        if min_selection is None or max_selection is None:
            min_selection = df[column_name] >= wave_min
            max_selection = df[column_name] <= wave_max

            return df[column_name].loc[(min_selection & max_selection)], min_selection, max_selection
        return df[column_name].loc[(min_selection & max_selection)]

    def get_jar_reference(self, x_column_selection, min_selection, max_selection):
        # Read jar reference data and apply wave range
        jar_read = self.read_jar_reference()
        jar_data = pd.concat([x_column_selection, self.apply_wave_range(jar_read, jar_read.columns[1],
                                                                        min_selection, max_selection)])

        jar_x_column_name = jar_data.columns[0]
        jar_data_column_name = jar_data.columns[1]

        # Calculate baseline and subtract from intensity. Add to jar_data DataFrame
        jar_intensity = jar_data[jar_data_column_name].to_numpy()
        jar_baseline = Algorithms.arpls(jar_intensity[jar_data_column_name].to_numpy())
        jar_corrected = jar_intensity - jar_baseline
        jar_data["jar_corrected"] = jar_corrected

        # Apply user-selected range for jar peak to x and intensity arrays and calculate area underneath the curve
        jar_min_selection = jar_data[jar_x_column_name] >= jar_scaling_range[0]
        jar_max_selection = jar_data[jar_x_column_name] <= jar_scaling_range[1]
        jar_corrected_ranged = jar_data["jar_corrected"].loc[(jar_min_selection & jar_max_selection)].to_numpy()
        jar_corrected_ranged_x = jar_data[jar_x_column_name].loc[(jar_min_selection & jar_max_selection)].to_numpy()
        jar_corrected_ranged_area = np.trapz(y=jar_corrected_ranged, x=jar_corrected_ranged_x)

        return jar_data, jar_corrected_ranged_x, jar_corrected_ranged_area, jar_min_selection, jar_max_selection

    def process_data(self, df: pd.DataFrame, current_file: str, head: str):
        x_column_name = df.columns[0]

        x_column_selection, min_selection, max_selection = self.apply_wave_range(df, x_column_name)

        output_df = pd.DataFrame()
        output_df[x_column_name] = x_column_selection

        if jar_correction:
            jar_data, jar_corrected_ranged_x, jar_corrected_ranged_area, jar_min_selection, jar_max_selection = \
                self.get_jar_reference(x_column_selection, min_selection, max_selection)

        for column_name in df.columns:
            if column_name == x_column_name:
                continue
            try:
                if test_row_index != '' and column_name != df.columns[test_row_index]:
                    continue
            except IndexError:
                pass

            intensity = self.apply_wave_range(df, column_name, min_selection, max_selection)

            if jar_correction:
                intensity_baseline_corrected, _, _ = self.add_baseline_diff(pd.DataFrame(intensity, columns=["y"]),
                                                                            "y", pd.DataFrame())
                # Calculate area underneath intensity curve in (jar) scaling range
                data_ranged = intensity_baseline_corrected.loc[(jar_min_selection & jar_max_selection)].to_numpy()
                data_ranged_area = np.trapz(y=data_ranged, x=jar_corrected_ranged_x)

                # Calculate scaling factor for jar curve and apply to jar curve
                jar_scaled = (data_ranged_area / jar_corrected_ranged_area) * jar_data["jar_corrected"]

                # Subtract jar reference intensity from intensity
                intensity = intensity - jar_scaled

            data = pd.concat([x_column_selection, intensity], axis='columns')
            data = data.reset_index(drop=True)

            pprint(data)

            output_df, baseline_diff, unscaled_corrected = self.add_baseline_diff(data, data.columns[1],
                                                                                  output_df, data.columns[0])

            baseline = pd.DataFrame()
            baseline['baseline'] = baseline_diff

            output_df = output_df.set_index(x_column_selection)
            baseline = baseline.set_index(x_column_selection)
            intensity = pd.DataFrame(intensity).set_index(x_column_selection)

            if plot_data:  # will plot every set of data if this option is enabled
                try:
                    plt.plot(intensity, color="blue", label="original")
                    plt.plot(baseline['baseline'], color="red", label="baseline")
                    plt.plot(unscaled_corrected, color="green", label="baseline corrected")

                    plt.xlabel(x_column_name)
                    plt.ylabel("intensity")
                    plt.legend(loc='upper right')
                    plt.title(column_name)

                    plt.show()
                except KeyError:
                    print(f'[{current_file}] failed plotting')

        WriteDFToFile(output_df, current_file[:-4] + '.dat', head=head, sep=dat_file_separator)


if __name__ == '__main__':
    BackgroundCorrection()
