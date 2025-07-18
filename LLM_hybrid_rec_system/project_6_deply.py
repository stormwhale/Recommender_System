import zipfile
import numpy as np
import pandas as pd
import json
import requests
import io
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import linear_kernel
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import col
from pyspark.sql.functions import explode


def prepare_dataset(local_path="df_reduce.csv"):
    if os.path.exists(local_path):
        print(f"{local_path} already exists. Skipping download.")
        return

    print("Downloading and preparing dataset...")
    url = "https://files.grouplens.org/datasets/movielens/ml_belief_2024_data_release_2.zip"
    response = requests.get(url)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
        with zip_file.open("data_release/movies.csv") as f:
            df_movie = pd.read_csv(f)
        with zip_file.open("data_release/user_rating_history.csv") as f:
            user_rating_df = pd.read_csv(f)

    df_ratings = user_rating_df.drop("tstamp", axis=1)
    df_com = df_ratings.merge(df_movie, on="movieId", how="left")

    # Clean data
    df_com_cleaned = df_com[
        (df_com["rating"] >= 0) & (~df_com["rating"].isna()) & (~df_com["title"].isna())
    ]

    # Reduce dataset
    df_reduce = df_com_cleaned.sample(frac=0.5, random_state=42)
    df_reduce.to_csv(local_path, index=False)

    print("Dataset prepared and saved to:", local_path)


class HybridRecommender:
    def __init__(self, df_csv):
        self.spark = SparkSession.builder.getOrCreate()
        self.py_df_full = self.spark.read.csv(df_csv, header=True, inferSchema=True)
        self.py_df = self.py_df_full.select("userId", "movieId", "rating")
        self.df_reduce = self.py_df_full.toPandas()

        # Train ALS model
        als = ALS(
            maxIter=10,
            rank=10,
            regParam=0.1,
            userCol="userId",
            itemCol="movieId",
            ratingCol="rating",
            nonnegative=True,
            coldStartStrategy="drop",
            seed=42,
        )
        self.als_model = als.fit(self.py_df)

        # Get ALS predictions
        rec_list = self.als_model.recommendForAllUsers(100)
        self.rec_list = rec_list.select(
            "userId", explode("recommendations").alias("rec")
        ).select(
            "userId",
            col("rec.movieId").alias("movieId"),
            col("rec.rating").alias("rating"),
        )

        self.rec_unrated = self.rec_list.join(
            self.py_df, on=["userId", "movieId"], how="left_anti"
        )

        # Prepare metadata
        self.movie_meta = self.py_df_full.select(
            "movieId", "title", "genres"
        ).dropDuplicates(
            ["movieId"]
        )  # Making sure the metadata has only unique movie titles and IDs

        # Prepare TF-IDF for content-based
        df_movies_unique = (
            self.df_reduce[["movieId", "title", "genres"]]
            .drop_duplicates("movieId")
            .reset_index(drop=True)
        )
        df_movies_unique["genres"] = df_movies_unique["genres"].fillna("")
        df_movies_unique["title"] = df_movies_unique["title"].fillna("")
        df_movies_unique["genres_clean"] = df_movies_unique["genres"].str.replace(
            "|", " ", regex=False
        )
        df_movies_unique["title_clean"] = (
            df_movies_unique["title"]
            .str.lower()
            .str.replace(r"[^a-z0-9\s]", "", regex=True)
        )
        df_movies_unique["features"] = (
            df_movies_unique["title_clean"]
            + " "
            + df_movies_unique["genres_clean"].str.lower()
        ).str.strip()

        self.df_movies_unique = df_movies_unique

        count_vec = CountVectorizer()
        self.count_matrix = count_vec.fit_transform(df_movies_unique["features"])
        self.cos_sim = linear_kernel(
            self.count_matrix
        )  # Linear kernel is used here to improve efficiency

        self.movie_id_to_index = pd.Series(
            df_movies_unique.index, index=df_movies_unique["movieId"]
        ).to_dict()

    def get_user_movie_vectors_for_prompt(self, user_id, movie_ids):
        # Item vectors:
        item_factors = self.als_model.itemFactors
        item_vectors = {}
        for mid in movie_ids:
            item_vectors[mid] = item_factors.filter(col("id") == mid).collect()[0][
                "features"
            ]

        item_vectors_str = {
            k: [round(x, 3) for x in v] for k, v in item_vectors.items()
        }
        item_vectors_str = json.dumps(item_vectors_str)

        # user vectors:
        user_vector = self.als_model.userFactors.filter(col("id") == user_id).collect()[
            0
        ]["features"]
        user_vector_str = json.dumps([round(x, 3) for x in user_vector])

        return {"user_vectors": user_vector_str, "item_vectors": item_vectors_str}

    def hybrid_recommender_for_user(self, user_id, top_n=5, alpha=0.5):
        user_rec = self.rec_unrated.filter(col("userId") == user_id).sort(
            "rating", ascending=False
        )
        user_rec_full = user_rec.join(self.movie_meta, on="movieId", how="left")
        user_rec_full_df = user_rec_full.toPandas()

        user_liked = self.df_reduce[
            (self.df_reduce["userId"] == user_id) & (self.df_reduce["rating"] >= 4)
        ]
        user_liked_movie_ids = user_liked["movieId"].tolist()
        user_liked_indices = [
            self.movie_id_to_index[mid]
            for mid in user_liked_movie_ids
            if mid in self.movie_id_to_index
        ]

        def compute_content_score_by_mid(mid):
            if mid not in self.movie_id_to_index or not user_liked_indices:
                return 0.0
            idx = self.movie_id_to_index[mid]
            return self.cos_sim[idx, user_liked_indices].mean()

        user_rec_full_df["cb_score"] = user_rec_full_df["movieId"].apply(
            compute_content_score_by_mid
        )
        user_rec_full_df["hybrid_score"] = (user_rec_full_df["cb_score"] * alpha) + (
            (1 - alpha) * user_rec_full_df["rating"]
        )

        return user_rec_full_df.sort_values(by="hybrid_score", ascending=False).head(
            top_n
        )

    def content_based_recommendations(self, user_id, top_n=5):
        import pandas as pd

        df = self.df_reduce
        df_movies = self.df_movies_unique.copy()

        user_liked = df[(df["userId"] == user_id) & (df["rating"] >= 4)]
        user_liked_movie_ids = user_liked["movieId"].tolist()
        user_liked_indices = [
            self.movie_id_to_index[mid]
            for mid in user_liked_movie_ids
            if mid in self.movie_id_to_index
        ]

        def compute_content_score_by_mid(mid):
            if mid not in self.movie_id_to_index or not user_liked_indices:
                return 0.0
            idx = self.movie_id_to_index[mid]
            return self.cos_sim[idx, user_liked_indices].mean()

        df_movies["cb_score"] = df_movies["movieId"].apply(compute_content_score_by_mid)
        df_movies = df_movies[~df_movies["movieId"].isin(user_liked_movie_ids)]

        return (
            df_movies[["movieId", "title", "features", "cb_score"]]
            .sort_values(by="cb_score", ascending=False)
            .head(top_n)
        )

    def cf_recommendations(self, user_id, top_n=5):
        user_rec = self.rec_unrated.filter(col("userId") == user_id).sort(
            "rating", ascending=False
        )
        user_rec_full = user_rec.join(self.movie_meta, on="movieId", how="left")
        return (
            user_rec_full.select("movieId", "title", "genres", "rating")
            .limit(top_n)
            .toPandas()
        )

    def llm_explain(
        self, user_id, top_n=5, alpha=0.5, model="gemma3:latest", include_content=True
    ):
        import ollama

        # get recommendation results and metadata:
        top_recs = self.hybrid_recommender_for_user(user_id, top_n, alpha)
        movie_ids = top_recs["movieId"].to_list()
        titles = top_recs["title"].to_list()
        genres = top_recs["genres"].to_list()
        cb_score = top_recs["cb_score"].to_list()

        # get latent factors for user and movies:
        latent_factors = self.get_user_movie_vectors_for_prompt(user_id, movie_ids)

        # Movie titles and genres that the user had liked in the past:
        liked_titles = self.df_reduce[
            (self.df_reduce["userId"] == user_id) & (self.df_reduce["rating"] >= 4)
        ]

        # Construct the prompt:
        movie_info = ""
        if include_content:
            movie_info = "\n".join(
                [
                    f"- {title} ({genre}) {cb_score}"
                    for title, genre, cb_score in zip(titles, genres, cb_score)
                ]
            )
        prompt = f"""
        You are a movie recommendation expert. A user has been analyzed using collaborative filtering and content-based filtering. 
        Here is the user's latent factor vector:
        {latent_factors}
        
        The top recommended movies for this user are:
        {movie_info}
        
        The user had liked these top_N movies from the past with title and genre:
        {liked_titles}
        
        which weights how much content-based filtering influences the final recommendation, is:
        {alpha}
        
        Explain the following:
        1) The user's movie genre preferences
        2) how the hybrid factor weights on the recommendations
        3) why these movies might be recommended to this user, referencing the vectors or user's past liked genres if relevant.
        4) Would you recommend any other movies different from the ones listed, analyzing from all provided data.
        
        """
        client = ollama.Client()
        response = client.generate(model=model, prompt=prompt)
        return response.response
