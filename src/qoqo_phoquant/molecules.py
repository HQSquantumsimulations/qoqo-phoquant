"""Python functions for loading treated molecular data for GBS."""

# Copyright © 2024 HQS Quantum Simulations GmbH. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
#  distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#  KIND, either express
# or implied. See the License for the specific language governing permissions
#  and limitations under
# the License.

import json
import numpy as np


class molecule:
    """Molecular information for computing vibronic spectrum with GBS."""

    def __init__(self, file_path: str) -> None:
        """Initialise molecule class instance.

        Args:
            file_path: path to json file with relevant information

        Raises:
            FileNotFoundError: Data file does not exist
        """
        # Read .json with data
        try:
            with open(file_path) as my_file:
                data_dict = json.load(my_file)
                # Parse the data
                self.U1 = np.array(data_dict["U1"])
                self.U2 = np.array(data_dict["U2"])
                self.s = np.array(data_dict["s"])
                self.alpha = np.array(data_dict["alpha"])
                self.E_vertical = float(data_dict["E_vertical"])
                self.freq_ini = np.array(data_dict["freq_ini"])
                self.freq_fin = np.array(data_dict["freq_fin"])
        except FileNotFoundError:
            print("Data file does not exist.")
