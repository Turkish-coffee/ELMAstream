import polars as pl
import pandas as pd
from typing import Dict, Text
import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs
from config import DATA

# len df : 100_480_507 data
df : pl.DataFrame = pl.read_parquet(source=DATA, columns=['movie_id','user_id'])

unique_movie_titles = np.unique(list(df['movie_id']))
unique_user_ids = np.unique(list(df['user_id']))

# divide data into smaller chunks of data to not overload the RAM 
# and allow computations
df = df.iter_slices(n_rows=1_000_000) # from this line df is an iterator


for partition in df:
  partition = partition.to_dict(as_series=False)

  # Load ratings data
  ratings = tf.data.Dataset.from_tensor_slices(partition)
  
  # Load movies data
  movies = tf.data.Dataset.from_tensor_slices(partition['movie_id'])

  ratings = ratings.map(lambda x: {
      "movie_id": x["movie_id"],
      "user_id": x["user_id"],
  })

  #shuffled = ratings.shuffle(1_000_000, seed=42, reshuffle_each_iteration=False)
  shuffled = ratings

  train = shuffled.take(900_000)
  test = shuffled.skip(900_000).take(100_000)

  movie_titles = movies.batch(1_024)
  user_ids = ratings.batch(10_000).map(lambda x: x["user_id"])

  #unique_movie_titles = np.unique(np.concatenate(list(movie_titles),dtype=np.int32))
  #unique_user_ids = np.unique(np.concatenate(list(user_ids),dtype=np.int32))

  embedding_dimension = 64

  user_model = tf.keras.Sequential([
    tf.keras.layers.IntegerLookup(
        vocabulary=unique_user_ids, mask_token=None),
    # We add an additional embedding to account for unknown tokens.
    tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
  ])

  movie_model = tf.keras.Sequential([
    tf.keras.layers.IntegerLookup(
        vocabulary=unique_movie_titles, mask_token=None),
    tf.keras.layers.Embedding(len(unique_movie_titles) + 1, embedding_dimension)
  ])


  metrics = tfrs.metrics.FactorizedTopK(
    candidates=movies.batch(512).map(movie_model)
  )


  task = tfrs.tasks.Retrieval(
    metrics=metrics
  )


  class MovielensModel(tfrs.Model):

    def __init__(self, user_model, movie_model):
      super().__init__()
      self.movie_model: tf.keras.Model = movie_model
      self.user_model: tf.keras.Model = user_model
      self.task: tf.keras.layers.Layer = task
      

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
      # We pick out the user features and pass them into the user model.
      user_embeddings = self.user_model(features["user_id"])
      # And pick out the movie features and pass them into the movie model,
      # getting embeddings back.
      positive_movie_embeddings = self.movie_model(features["movie_id"])

      # The task computes the loss and the metrics.
      return self.task(user_embeddings, positive_movie_embeddings)


  model = MovielensModel(user_model, movie_model)
  model.compile(optimizer=tf.keras.optimizers.legacy.Adagrad(learning_rate=0.1))


  cached_train = train.shuffle(900_000).batch(65536).cache()
  cached_test = test.batch(4096).cache()

  model.fit(cached_train, epochs=1)


  model.evaluate(cached_test, return_dict=True)


  # Create a model that takes in raw query features, and
  index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
  # recommends movies out of the entire movies dataset.
  index.index_from_dataset(
    tf.data.Dataset.zip((movies.batch(100), movies.batch(100).map(model.movie_model)))
  )

  # Get recommendations.
  _, titles = index(tf.constant([42]))
  print(f"Recommendations for user 42: {titles[0, :3]}")
