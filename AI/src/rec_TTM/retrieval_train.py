import polars as pl
import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs
from config import DATA
from retrieval_model import build_retrieval_system

# len df : 100_480_507 data
print("fetching data ...")
df : pl.DataFrame = pl.read_parquet(source=DATA, columns=['movie_id','user_id'])
# storing vocabulary (user / item) to define embbedings later on 
unique_movie_titles = np.unique(list(df['movie_id']))
unique_user_ids = np.unique(list(df['user_id']))
# divide data into smaller chunks of data to not overload the RAM 
# and allow computations
df = df.iter_slices(n_rows=100_000) # from this line df is an iterator
print("data fetched successfully")
# defining model outside of the training loop to avoid overwite of the 
# learnt embbedings. -> setting the self.task inside the loop

print("building retrieval model ...")
model = build_retrieval_system( embedding_dim=64,
                                query_vocabulary=unique_user_ids,
                                candidate_vocabulary= unique_movie_titles)

model.set_task(candidates=tf.data.Dataset.from_tensor_slices(unique_movie_titles))
model.compile(optimizer=tf.keras.optimizers.legacy.Adagrad(learning_rate=0.1))

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

  model.fit(cached_train, epochs=1, verbose=2)
  model.evaluate(cached_test, return_dict=True, verbose=2)

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