import os
from deeplogger import DATA_DIR
from deeplogger.common_helpers import create_directory
from pooch import __version__, check_version, HTTPDownloader, retrieve
import zipfile

VALTER_boreholes = {'CB1': 'https://polybox.ethz.ch/index.php/s/sJgS2Iw4IcPBRa3/download',
                    'CB3': 'https://polybox.ethz.ch/index.php/s/VBwbF3EUVih9woU/download',
                    'MB5': 'https://polybox.ethz.ch/index.php/s/gn4PGXzbDggue4c/download',
                    'MB7': 'https://polybox.ethz.ch/index.php/s/gs9GaLw01h933ok/download',
                    'MB8': 'https://polybox.ethz.ch/index.php/s/LJ6NpVUyXNamVAl/download',
                    'ST1': 'https://polybox.ethz.ch/index.php/s/MGvZg0XPqJChCXS/download',
                    'ST2': 'https://polybox.ethz.ch/index.php/s/2BYkCuKbrpt1H8i/download'}

metadata_file = 'https://polybox.ethz.ch/index.php/s/yHGFutEDPp26X7X/download'

def download_VALTER_borehole(borehole_name: str,
                             additional_directory=None):
    '''Download the data of a borehole from the VALTER project. 
    The data is downloaded from the ETH Polybox and unzipped in the data folder.
    The function returns the folder path and the list of files in the folder.
    Args:
        additional_directory: include the data in a subdirectory of the data folder.
        borehole_name (str): name of the borehole to download.
    Returns:
        folder_path (str): path of the folder containing the data.
        files (list): list of files in the folder.
    '''
    # check if borehole_name is in the list of boreholes
    if borehole_name not in VALTER_boreholes.keys():
        raise ValueError(f"borehole_name should be one of {list(VALTER_boreholes.keys())}")
    url = VALTER_boreholes[borehole_name]
    if additional_directory is not None:
        data_output_directory = DATA_DIR + additional_directory + os.sep
        from deeplogger.common_helpers import create_directory
        create_directory(DATA_DIR)
    else:
        data_output_directory = DATA_DIR
    file_path = data_output_directory + borehole_name + ".zip"
    url = url.format(check_version(__version__, fallback="main"))
    downloader = HTTPDownloader()
    downloader(url=url, output_file=file_path, pooch=None)

    while not os.path.exists(data_output_directory + borehole_name + ".zip"):
        print('File being downloaded')
    file_name = os.path.abspath(file_path)  # get full path of files
    zip_ref = zipfile.ZipFile(file_name)  # create zipfile object
    zip_ref.extractall(data_output_directory)  # extract file to dir
    os.remove(file_path)  # remove zip file
    # get list of files in the directory
    files = os.listdir(data_output_directory + borehole_name)
    return data_output_directory + borehole_name + os.sep, files

def download_file(url, file_name, additional_directory=None):
    '''Download a file from a URL and save it in the data folder.
    Args:
        url (str): URL of the file to download.
        file_name (str): name of the file to save.
    Returns:
        file_path (str): path of the downloaded file.
    '''
    if additional_directory is not None:
        data_output_directory = DATA_DIR + additional_directory + os.sep
        from deeplogger.common_helpers import create_directory
        create_directory(DATA_DIR)
    else:
        data_output_directory = DATA_DIR
    file_path = data_output_directory + file_name
    url = url.format(check_version(__version__, fallback="main"))
    downloader = HTTPDownloader()
    downloader(url=url, output_file=file_path, pooch=None)
    return file_path

create_directory(DATA_DIR + 'Bedretto_Input_HS')

for borehole in VALTER_boreholes.keys():
    download_VALTER_borehole(borehole,
                             additional_directory='Bedretto_Input_HS')
    print(f'{borehole} downloaded')

download_file(metadata_file,
              'file_informations.xlsx',
              additional_directory='Bedretto_Input_HS')