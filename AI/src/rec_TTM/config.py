import os

BASE_DIR = os.path.abspath(os.path.join(__file__, '../../../data/netflix_recommendations'))  
RATINGS_FILES = [
    os.path.join(
        BASE_DIR ,
        'combined_data_' + str(i) + '.txt'
        ) for i in range(1,5)
]

DATA = os.path.join(BASE_DIR,'output.parquet')

