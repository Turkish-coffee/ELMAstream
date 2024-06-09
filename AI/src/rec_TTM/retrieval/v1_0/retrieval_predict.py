import os
from typing import List
import tensorflow as tf
import polars as pl
import tensorflow_recommenders as tfrs
from rec_TTM.retrieval.v1_0.config import DATA, BASE_DIR, BASE_SAVE_DIR
from rec_TTM.utils.preprocessing import load_data, map_movie_id_to_titles
from rec_TTM.retrieval.v1_0.retrieval_model import RetrievalModel, load_retrieval_system, build_retrieval_system

'''
    the following function will perform a prediciton of
    the model on a determined candidate sample over a
    specific user, and thus, return a list of candidates
'''
def predict_user_candidates(model : RetrievalModel, movies : tf.data.Dataset, user_id : int) -> List:
    index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
    index.index_from_dataset(
        tf.data.Dataset.zip((
                movies.batch(100),
                movies.batch(100).map(model.movie_model)
            )
        )
    )

    # Get recommendations.
    _, recommended_movie_ids = index(tf.constant([user_id]))
    print(type(recommended_movie_ids[0, :3].numpy().tolist()))
    movies = map_movie_id_to_titles(movie_ids=recommended_movie_ids[0, :3].numpy().tolist(),
                                    df=pl.read_csv(
                                        os.path.join(BASE_DIR,'movie_titles.csv'),
                                        new_columns=['movie_id','year','movie_title'],
                                        truncate_ragged_lines=True,
                                        infer_schema_length=10000,
                                        encoding='ISO-8859-1',
                                        has_header=False
                                        )
                                    )

    print(f"Recommendations for user {user_id}: {movies[:]}")
    

if __name__ == "__main__":

    _ , unique_movie_titles, unique_user_ids = load_data()

    model = build_retrieval_system( embedding_dim=64,
                                    query_vocabulary=unique_user_ids,
                                    candidate_vocabulary= unique_movie_titles )
    load_retrieval_system(model=model, model_path=os.path.join(BASE_SAVE_DIR,'retrieval','v1_0','retrieval_model'))
    movies = tf.data.Dataset.from_tensor_slices(unique_movie_titles)
    predict_user_candidates(model=model, movies=movies, user_id=42)