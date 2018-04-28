from datetime import datetime

from constants import Constants


class Query(object):
    """
    algorithm_name: string
    start_date: int (timestamp)
    end_date: int (timestamp)
    params: dict
    resolution: string
    key_selector: string
    sample: float
    """
    def __init__(self, algorithm_name, start_date, end_date, params, resolution, key_selector,
                 sample):
        self.algorithm_name = algorithm_name
        self.start_date = start_date
        self.end_date = end_date
        self.params = params
        self.resolution = resolution
        self.key_selector = key_selector
        self.sample = sample

    def __repr__(self):
        start_date_nice = datetime.fromtimestamp(self.start_date).strftime('%Y-%m-%d %H:%M:%S')
        end_date_nice = datetime.fromtimestamp(self.end_date).strftime('%Y-%m-%d %H:%M:%S')

        return "Query(algorithm_name = {}, start_date = '{}', end_date = '{}', params = {}, " \
               "resolution = {}, key_selector = '{}', sample = {})".format(self.algorithm_name,
                                                                           start_date_nice,
                                                                           end_date_nice,
                                                                           self.params,
                                                                           self.resolution,
                                                                           self.key_selector,
                                                                           self.sample)

    def encode(self):
        algorithm_encoding = self.encode_algorithm(self.algorithm_name)
        start_date_encoding = self.start_date
        end_date_encoding = self.end_date
        first_window_encoding, second_window_encoding = self.encode_params(self.params)
        resolution_encoding = self.encode_resolution(self.resolution)
        key_selector_encoding = self.encode_key_selector(self.key_selector)
        sample_encoding = self.sample

        return [
            algorithm_encoding,
            start_date_encoding,
            end_date_encoding,
            first_window_encoding,
            second_window_encoding,
            resolution_encoding,
            key_selector_encoding,
            sample_encoding
        ]

    # TODO: Move encoding functions to another file
    def encode_algorithm(self, algorithm):
        return algorithm.value

    def encode_resolution(self, resolution):
        return resolution.value

    def encode_key_selector(self, key_selector):
        if key_selector in Constants.LEVEL_1_LOCATIONS:
            return Constants.LEVEL_1_LOCATIONS.index(key_selector)
        elif key_selector in Constants.LEVEL_2_LOCATIONS:
            return Constants.LEVEL_2_LOCATIONS.index(key_selector)
        return Constants.LEVEL_3_LOCATIONS.index(key_selector)

    def encode_params(self, params):
        first_window = 0
        if "first_window" in params:
            first_window = params.first_window

        second_window = 0
        if "second_window" in params:
            second_window = params.second_window

        return first_window, second_window
