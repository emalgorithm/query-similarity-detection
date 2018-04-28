import pandas as pd

from queries_generation import generate_density_queries_pair, generate_mobility_queries_pair


def generate_data(size=1000):
    """
    Generate datasets containing pairs of density and mobility queries
    """
    data = []

    # Generate similar mobility queries pair
    for i in range(size):
        similar = True
        q1, q2 = generate_density_queries_pair(similar=similar)
        data.append([q1.encode(), q2.encode(), int(similar)])

    # Generate not similar mobility queries pair
    for i in range(size):
        similar = False
        q1, q2 = generate_density_queries_pair(similar=similar)
        data.append([q1.encode(), q2.encode(), int(similar)])

    # Generate similar density queries pair
    for i in range(size):
        similar = True
        q1, q2 = generate_mobility_queries_pair(similar=similar)
        data.append([q1.encode(), q2.encode(), int(similar)])

    # Generate not similar density queries pair
    for i in range(size):
        similar = False
        q1, q2 = generate_mobility_queries_pair(similar=similar)
        data.append([q1.encode(), q2.encode(), int(similar)])

    # Create pandas dataframe
    df = pd.DataFrame(data=data, columns=['query_1', 'query_2', 'similar'])

    # Shuffle dataset
    df = df.sample(frac=1).reset_index(drop=True)

    # Write to csv
    df.to_csv("data/density_mobility_queries.csv", index=False)
