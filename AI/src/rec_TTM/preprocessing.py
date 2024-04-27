import os
from typing import Dict, List
from config import BASE_DIR, RATINGS_FILES, DATA


'''
this function will take the raw data provided by netflix and
transform it to a lightweight parquet file which will be easier
to load and manipulate troughout the ml workflow.
'''

def process_data(files: List[str]) -> None:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    import datetime

    print("starting preproccessing ...")
    data = []
    current_movie = None
    for file in files: 
        print(f"preproccessing {file} ...")
        with open(file, 'r') as f:
            review_index = 0
            for line in f:
                if line.strip().endswith(':'):
                    current_movie = line.strip()[:-1]  # Remove the ':' character
                else:
                    review_index += 1
                    movie_data = line.strip().split(',')
                    data.append({
                                "review_id": review_index,
                                "movie_id": int(current_movie),
                                "user_id":int(movie_data[0]),
                                "rating": float(movie_data[1]),
                                "year": datetime.datetime.strptime(movie_data[2], '%Y-%m-%d').year
                            }
                        )
    # Convert array of dictionaries to DataFrame
    df = pd.DataFrame(data)
    # Convert DataFrame to PyArrow Table
    table = pa.Table.from_pandas(df)
    # Write PyArrow Table to Parquet file
    output_file = 'output.parquet'
    pq.write_table(table, os.path.join(BASE_DIR, output_file))
    print(f"Parquet file '{output_file}' created successfully.")
    print(f"Parquet file '{output_file}' sotred to {BASE_DIR}.")


"""
this function is in charge of loading data in a more efficient
way, using spark to create multiple homogenous partitions from
the huge base parquet file
"""
def load_data() -> None:
    pass

if __name__ == "__main__":
    #process_data(RATINGS_FILES)
    load_data()
