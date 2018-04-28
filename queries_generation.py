from query import Query
import random

from constants import Resolution, AlgorithmName, Constants


def generate_density_queries_pair(similar=False):
    """
    Generates two density queries which are not similar to each other.
    """
    q1 = generate_random_density_query()
    if similar:
        q2 = generate_similar_density_query(q1)
    else:
        q2 = generate_not_similar_density_query(q1)

    return q1, q2


def generate_mobility_queries_pair(similar=False):
    """
    Generates two density queries which are not similar to each other.
    """
    q1 = generate_random_mobility_query()
    if similar:
        q2 = generate_similar_mobility_query(q1)
    else:
        q2 = generate_not_similar_mobility_query(q1)

    return q1, q2


def generate_not_similar_density_query(q1):
    q2 = generate_random_density_query()

    while are_density_queries_similar(q1, q2):
        q2 = generate_random_density_query()

    return q2


def generate_not_similar_mobility_query(q1):
    q2 = generate_random_mobility_query()

    while are_mobility_queries_similar(q1, q2):
        q2 = generate_random_mobility_query()

    return q2


def are_density_queries_similar(q1, q2):
    return are_timestamps_similar(q1.start_date, q2.start_date) and \
           are_timestamps_similar(q1.end_date, q2.end_date) and \
           q1.resolution == q2.resolution and \
           q1.key_selector == q2.key_selector


def are_mobility_queries_similar(q1, q2):
    return are_timestamps_similar(q1.start_date, q2.start_date) and \
           are_timestamps_similar(q1.end_date, q2.end_date) and \
           are_timestamps_similar(q1.params["first_window"], q2.params["first_window"]) and \
           are_timestamps_similar(q1.params["second_window"], q2.params["second_window"]) and \
           q1.resolution == q2.resolution and \
           q1.key_selector == q2.key_selector


def are_timestamps_similar(t1, t2):
    return abs(t1 - t2) < 60 * 60


def generate_random_density_query():
    algorithm_name = AlgorithmName.DENSITY
    start_date = generate_random_timestamp()
    end_date = generate_random_timestamp(min_timestamp=start_date)
    params = {}
    resolution = random.choice(Constants.LOCATION_LEVELS)
    key_selector = generate_random_location(resolution)
    sample = random.uniform(0, 1)

    return Query(algorithm_name=algorithm_name, start_date=start_date, end_date=end_date,
                 params=params, resolution=resolution, key_selector=key_selector, sample=sample)


def generate_similar_density_query(query):
    """
        Generates a density query similar to the given density one.
        Two density queries are similar when:
        - resolution = resolution’
        - keySelector = keySelector’
        - |startDate - startDate’| < 60 minutes
        - |endDate - endDate’| < 60 minutes
    """
    similar_start_date = generate_similar_timestamp(query.start_date)
    similar_end_date = generate_similar_timestamp(query.end_date, min_timestamp=similar_start_date)
    different_sample = random.uniform(0, 1)

    return Query(algorithm_name=query.algorithm_name, start_date=similar_start_date,
                 end_date=similar_end_date, params=query.params, resolution=query.resolution,
                 key_selector=query.key_selector, sample=different_sample)


def generate_similar_mobility_queries_pair():
    """
        Generates two mobility queries which are similar to each other.
        Two density queries are similar when:
        - resolution = resolution’
        - keySelector = keySelector’
        - |startDate - startDate’| < 60 minutes
        - |first_window - first_window’| < 60 minutes
        - |endDate - endDate’| < 60 minutes
        - |second_window - second_window’| < 60 minutes
    """
    q1 = generate_random_mobility_query()
    q2 = generate_similar_mobility_query(q1)

    return q1, q2


def generate_random_mobility_query():
    algorithm_name = AlgorithmName.MOBILITY
    start_date = generate_random_timestamp()
    end_date = generate_random_timestamp(min_timestamp=start_date)
    first_window = generate_random_timestamp(min_timestamp=start_date, max_timestamp=end_date)
    second_window = generate_random_timestamp(min_timestamp=first_window, max_timestamp=end_date)
    params = {
        "first_window": first_window,
        "second_window": second_window
              }
    resolution = random.choice(Constants.LOCATION_LEVELS)
    key_selector = generate_random_location(resolution) + "." + generate_random_location(resolution)
    sample = random.uniform(0, 1)

    return Query(algorithm_name=algorithm_name, start_date=start_date, end_date=end_date,
                 params=params, resolution=resolution, key_selector=key_selector, sample=sample)


def generate_similar_mobility_query(query):
    """
        Generates a mobility query similar to the given mobility query.
        Two mobility queries are similar when:
        - resolution = resolution’
        - keySelector = keySelector’
        - |startDate - startDate’| < 60 minutes
        - |endDate - endDate’| < 60 minutes
    """
    similar_start_date = generate_similar_timestamp(query.start_date)
    similar_end_date = generate_similar_timestamp(query.end_date, min_timestamp=similar_start_date)
    similar_first_window = generate_similar_timestamp(query.params["first_window"],
                                                      min_timestamp=similar_start_date,
                                                      max_timestamp=similar_end_date)
    similar_second_window = generate_similar_timestamp(query.params["second_window"],
                                                       min_timestamp=similar_first_window,
                                                       max_timestamp=similar_end_date)
    params = {
        "first_window": similar_first_window,
        "second_window": similar_second_window
              }
    different_sample = random.uniform(0, 1)

    return Query(algorithm_name=query.algorithm_name, start_date=similar_start_date,
                 end_date=similar_end_date, params=params, resolution=query.resolution,
                 key_selector=query.key_selector, sample=different_sample)


# def generate_similar_mobility_query_to_density_query(query):
#     """
#         Generates a mobility query similar to the given density query.
#         Two mobility queries are similar when:
#         - resolution = resolution’
#         - keySelector = keySelector’
#         # TODO
#     """
#     similar_start_date = generate_similar_timestamp(query.start_date)
#     similar_end_date = generate_similar_timestamp(query.end_date, min_timestamp=similar_start_date)
#     similar_first_window = generate_similar_timestamp(query.params["first_window"],
#                                                       min_timestamp=similar_start_date,
#                                                       max_timestamp=similar_end_date)
#     similar_second_window = generate_similar_timestamp(query.params["second_window"],
#                                                        min_timestamp=similar_first_window,
#                                                        max_timestamp=similar_end_date)
#     params = {
#         "first_window": similar_first_window,
#         "second_window": similar_second_window
#     }
#     different_sample = random.uniform(0, 1)
#
#     return Query(algorithm_name=query.algorithm_name, start_date=similar_start_date,
#                  end_date=similar_end_date, params=params, resolution=query.resolution,
#                  key_selector=query.key_selector, sample=different_sample)


def generate_random_timestamp(min_timestamp=Constants.MIN_TIMESTAMP, max_timestamp=Constants.MAX_TIMESTAMP):
    """
    Generate a random int timestamp between the provided timestamps
    """
    return random.randint(min_timestamp, max_timestamp)


def generate_similar_timestamp(timestamp, min_timestamp=Constants.MIN_TIMESTAMP,
                               max_timestamp=Constants.MAX_TIMESTAMP):
    """
    Generate a random int timestamp similar to the given timestamp and between the provided timestamps
    """
    # 60 minutes in second
    similarity_threshold = 60 * 60

    min_timestamp = max(min_timestamp, timestamp - similarity_threshold)
    max_timestamp = min(max_timestamp, timestamp + similarity_threshold)

    return generate_random_timestamp(min_timestamp, max_timestamp)


def generate_random_location(resolution):
    """
        Generate a random location for the given resolution
    """
    if resolution == Resolution.LOCATION_LEVEL_1:
        return random.choice(Constants.LEVEL_1_LOCATIONS)
    elif resolution == Resolution.LOCATION_LEVEL_2:
        return random.choice(Constants.LEVEL_2_LOCATIONS)
    return random.choice(Constants.LEVEL_3_LOCATIONS)
