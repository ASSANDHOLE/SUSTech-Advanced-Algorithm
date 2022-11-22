import os
import argparse

import numpy as np
from scipy.io import loadmat


def save_np_array(save_filename: str, data: np.ndarray) -> None:
    """
    Saves a numpy array to a binary file.

    Parameters
    ----------
    save_filename : str
        The filename to save the data to.
    data : np.ndarray
        The data to save.
    """
    with open(save_filename, 'wb') as f:
        f.write(data.tobytes(order='C'))


def convert_matlab_mat_file(filename: str, data_dir: str) -> None:
    """
    Converts a MATLAB .mat file to binary data file(s) that can be read by C++.

    Parameters
    ----------
    filename : str
        The filename of the .mat file to convert.
    data_dir : str
        The directory to save the converted data to.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f'File {filename} not found.')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    data = loadmat(filename)
    keys = []
    for key in data:
        if key.startswith('__'):
            continue
        keys.append((key, data[key].shape, data[key].dtype))
        save_np_array(os.path.join(data_dir, key + '.bin'), data[key])
        print(data[key].shape, data[key].dtype)

    # Save the keys to a file
    data_types = {
        np.uint8: 'ui8',
        np.uint16: 'ui16',
        np.uint32: 'ui32',
        np.uint64: 'ui64',
        np.int8: 'i8',
        np.int16: 'i16',
        np.int32: 'i32',
        np.int64: 'i64',
        np.float16: 'f16',
        np.float32: 'f32',
        np.float64: 'f64',
    }
    with open(os.path.join(data_dir, 'keys.txt'), 'w') as f:
        f.write('\n'.join([
            f'{key}:({" ".join([str(i) for i in shape])}):{data_types[dtype.type]}' for key, shape, dtype in keys
        ]))


def main():
    parser = argparse.ArgumentParser(
        description='Converts a MATLAB .mat file to binary data file(s) that can be read by C++.'
    )
    parser.add_argument('filename', type=str, help='The filename of the .mat file to convert.')
    parser.add_argument('data_dir', type=str, help='The directory to save the converted data to.')
    args = parser.parse_args()
    try:
        convert_matlab_mat_file(args.filename, args.data_dir)
        exit(0)
    except Exception as e:
        print(e)
        exit(1)


if __name__ == '__main__':
    main()
