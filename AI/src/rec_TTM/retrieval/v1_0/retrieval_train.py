import polars as pl
import tensorflow as tf
import tensorflow_recommenders as tfrs
from rec_TTM.retrieval.v1_0.config import RETRIEVAL_EPOCHS

'''
this function is a trainer wrapper :
it simplifies the call of training, taking the model
and the df artifacts in input. the segmentation of
df into smaller pieces allows the model to train
even if the entire data dosen't fit in memory.
once trained the model artifact is returned
'''
def train_model(df : pl.DataFrame, model : tfrs.Model) -> tfrs.Model:
  for index, partition in enumerate(df):
    #print(f"{index} / {len(df)} iteration: ") TODO: debug iterator
    print(f"{index} iteration : ")
    partition = partition.to_dict(as_series=False)

    # Load ratings data
    ratings = tf.data.Dataset.from_tensor_slices(partition)
    # Load movies data
    movies = tf.data.Dataset.from_tensor_slices(partition['movie_id'])

    ratings = ratings.map(lambda x: {
        "movie_id": x["movie_id"],
        "user_id": x["user_id"],
    })

    train = ratings.take(90_000)
    test = ratings.skip(90_000).take(10_000)
    
    cached_train = train.batch(8192).cache()
    cached_test = test.batch(1024).cache()

    model.fit(cached_train, epochs=RETRIEVAL_EPOCHS, verbose=2)
    model.evaluate(cached_test, return_dict=True, verbose=2)
  return model
"""
  # Create a model that takes in raw query features, and
  index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
  # recommends movies out of the entire movies dataset.
  index.index_from_dataset(
    tf.data.Dataset.zip((movies.batch(100), movies.batch(100).map(model.movie_model)))
  )

  # Get recommendations.
  _, titles = index(tf.constant([42]))
  print(f"Recommendations for user 42: {titles[0, :3]}")
"""