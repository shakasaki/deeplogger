# load pytorch sensors from a directory and check if there are any nans
from deeplogger import DATA_DIR
import os
from deeplogger.common_helpers import check_tensors_for_nans

data_directory = DATA_DIR + 'Bedretto_Output' + os.sep

nan_table = check_tensors_for_nans(data_directory)