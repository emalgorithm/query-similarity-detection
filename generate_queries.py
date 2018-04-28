from query import Query
import random

from constants import Resolution, AlgorithmName, Constants


def generate_similar_density_queries_pair():
    """
    Generates two density queries which are similar to each other.
    Two density queries are similar when:
    - resolution = resolution’
    - keySelector = keySelector’
    - |startDate - startDate’| < 60 minutes
    - |endDate - endDate’| < 60 minutes
    """
    q1 = generate_random_density_query()
    q2 = generate_similar_density_query(q1)

    return q1, q2


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
        Generates a query similar to the given one.
        Two density queries are similar when:
        - resolution = resolution’
        - keySelector = keySelector’
        - |startDate - startDate’| < 60 minutes
        - |endDate - endDate’| < 60 minutes
    """
    similar_start_date = generate_similar_timestamp(query.start_date)
    similar_end_date = generate_similar_timestamp(query.end_date)
    different_sample = random.uniform(0, 1)

    return Query(algorithm_name=query.algorithm_name, start_date=similar_start_date,
                 end_date=similar_end_date, params=query.params, resolution=query.resolution,
                 key_selector=query.key_selector, sample=different_sample)


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

# def generate_random_query():
#     startDate = generate_random_datetime()
#     endDate = generate_random_datetime(min_year=startDate.year)
#     algorithm = np.random.choice(['density', 'migration', 'commuting'])
#     params = {}
#     aggregationLevel = np.random.choice(['region', 'commune', 'antenna'])
#     aggregationValue = np.random.choice(['Dakar', 'London', 'Rome'])
#     sample = 1
#
#     query = Query(startDate, endDate, algorithm, params, aggregationLevel, aggregationValue, sample)
#     return query
#
# def generate_random_queries(n=100):
# 	return [generate_random_query() for i in range(n)]
#
# def generate_random_datetime(min_year=1900, max_year=datetime.now().year):
#     start = datetime(min_year, 1, 1, 00, 00, 00)
#     years = max_year - min_year + 1
#     end = start + timedelta(days=365 * years)
#
#     return start + (end - start) * random.random()
#
#
# print(generate_random_queries())