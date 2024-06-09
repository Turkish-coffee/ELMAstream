import tensorflow as tf
import tensorflow_recommenders as tfrs
from typing import Dict, Text
import numpy as np
class RetrievalModel(tfrs.Model):

    def __init__(self, user_model, movie_model, task):
      super().__init__()
      self.movie_model: tf.keras.Model = movie_model
      self.user_model: tf.keras.Model = user_model
      self.task: tfrs.tasks.Retrieval = task # task is set to None by default
      # so using model without setting it will throw an error, so make sure to 
      # set it

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
      # We pick out the user features and pass them into the user model.
      user_embeddings = self.user_model(features["user_id"])
      # And pick out the movie features and pass them into the movie model,
      # getting embeddings back.
      positive_movie_embeddings = self.movie_model(features["movie_id"])

      # The task computes the loss and the metrics. 
      # (set compute_metric to false to save time during execution)
      return self.task(user_embeddings, positive_movie_embeddings, compute_metrics=True)
    

'''
the following function is a wrapper whcih will take
in charge the creation of model instance 
'''
def build_retrieval_system( embedding_dim : int,
                            query_vocabulary: np.array,
                            candidate_vocabulary: np.array ) -> RetrievalModel:
  print("building retrieval model ...")
  user_model = tf.keras.Sequential([
    tf.keras.layers.IntegerLookup(
        vocabulary=query_vocabulary, mask_token=None),
    # We add an additional embedding to account for unknown tokens.
    tf.keras.layers.Embedding(len(query_vocabulary) + 1, embedding_dim)
  ])

  movie_model = tf.keras.Sequential([
    tf.keras.layers.IntegerLookup(
        vocabulary=candidate_vocabulary, mask_token=None),
    tf.keras.layers.Embedding(len(candidate_vocabulary) + 1, embedding_dim)
  ])

  task : tfrs.tasks = tfrs.tasks.Retrieval(
        metrics=tfrs.metrics.FactorizedTopK(
        candidates=tf.data.Dataset.from_tensor_slices(candidate_vocabulary)
        .batch(512).map(movie_model)
        )
      )
  
  return RetrievalModel(user_model=user_model, movie_model=movie_model, task=task)



'''
saves the model artifact to specified directory
'''
def save_retrieval_system(model:RetrievalModel, model_path : str) -> None:
  model.save_weights(model_path)
  print(f'model sucessfully saved to {model_path}')

'''
loads the model from the specified directory
'''
def load_retrieval_system(model : RetrievalModel, model_path : str) -> RetrievalModel:
  model.load_weights(model_path)
  return model