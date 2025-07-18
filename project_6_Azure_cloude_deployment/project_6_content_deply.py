import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import linear_kernel


print("content_based recommender starting...")


class contentBasedRec:
    def __init__(self, df_csv):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(base_dir, df_csv)
        self.df_csv = pd.read_csv(csv_path)

        # Prepare TF-IDF for content-based
        df_movies_unique = (
            self.df_csv[["movieId", "title", "genres"]]
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

    def content_based_recommendations(self, user_id, top_n=5):
        df = self.df_csv
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

    def get_user_list(self, top_n=10):
        return self.df_csv["userId"].head(top_n).tolist()

    def get_movie_list(self, top_n=10):
        return self.df_csv[["title", "genres"]].head(top_n).to_dict(orient="records")
