class Query(object):
    def __init__(self, algorithm_name, start_date, end_date, params, resolution, key_selector,
                 sample):
        self.algorithm_name = algorithm_name
        self.start_date = start_date
        self.end_date = end_date
        self.params = params
        self.resolution = resolution
        self.key_selector = key_selector
        self.sample = sample

    def encode(self):
        algorithm_encoding = self.encode_algorithm(self.algorithm_name)
        start_date_encoding = self.start_date.timestamp()
        end_date_encoding = self.end_date.timestamp()
        first_window_encoding, second_window_encoding = self.encode_params(self.params)
        resolution_encoding = self.encode_resolution(self.resolution)
        key_selector_encoding = self.encode_key_selector(self.key_selector)

        return [
            algorithm_encoding,
            start_date_encoding,
            end_date_encoding,
            first_window_encoding,
            second_window_encoding,
            resolution_encoding,
            key_selector_encoding
        ]

    # TODO: Move encoding functions to another file
    def encode_algorithm(self, algorithm):
        algorithms = ['density', 'mobility']
        return algorithms.index(algorithm)

    def encode_resolution(self, resolution):
        resolutions = ['location_level_1', 'location_level_2', 'location_level_3']
        return resolutions.index(resolution)

    def encode_key_selector(self, key_selector):
        # TODO: Hash?
        key_selectors = ['Dakar', 'London', 'Rome']
        return key_selectors.index(key_selector)

    def encode_params(self, params):
        return params.first_window.timestamp(), params.second_window.timestamp()


        
    