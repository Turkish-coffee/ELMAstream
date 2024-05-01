import tensorflow as tf
from preprocessing import load_data
from retrieval_model import build_retrieval_system
from retrieval_train import train_model

def main() -> None:
  df, unique_movie_titles, unique_user_ids = load_data()

  model = build_retrieval_system( embedding_dim=64,
                                  query_vocabulary=unique_user_ids,
                                  candidate_vocabulary= unique_movie_titles)

  model.set_task(candidates=tf.data.Dataset.from_tensor_slices(unique_movie_titles))
  model.compile(optimizer=tf.keras.optimizers.legacy.Adagrad(learning_rate=0.1))

  model = train_model(df=df, model=model)

if __name__ == "__main__":
    main()