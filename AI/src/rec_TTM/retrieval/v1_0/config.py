import os

BASE_DIR = os.path.abspath(os.path.join(__file__, '../../../../../data/netflix_recommendations'))  
BASE_SAVE_DIR = os.path.abspath(os.path.join(__file__, '../../../../../models/rec_TTM'))
RATINGS_FILES = [
    os.path.join(
        BASE_DIR ,
        'combined_data_' + str(i) + '.txt'
        ) for i in range(1,5)
]
DATA = os.path.join(BASE_DIR,'output.parquet')
RETRIEVAL_EPOCHS = 3