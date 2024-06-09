import os
import mlflow
import tensorflow as tf
from rec_TTM.utils.preprocessing import load_data
from rec_TTM.retrieval.v1_0.config import BASE_SAVE_DIR
from rec_TTM.retrieval.v1_0.retrieval_model import build_retrieval_system, save_retrieval_system
from rec_TTM.retrieval.v1_0.retrieval_train import train_model

def main() -> None:
  df, unique_movie_titles, unique_user_ids = load_data()
  model = build_retrieval_system( embedding_dim=64,
                                  query_vocabulary=unique_user_ids,
                                  candidate_vocabulary= unique_movie_titles)
  model.compile(optimizer=tf.keras.optimizers.legacy.Adagrad(learning_rate=0.1))
  model = train_model(df=df, model=model, epochs=1)
  save_retrieval_system(model=model, model_path=os.path.join(BASE_SAVE_DIR,'retrieval','v1_0','retrieval_model'))

if __name__ == "__main__":
    main()