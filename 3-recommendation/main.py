"""
See README.md for running instructions, examples and authors.
"""

from __future__ import annotations

import json
from typing import TypedDict, cast

import pandas as pd
from simple_term_menu import TerminalMenu
from sklearn.cluster import KMeans

CLUSTERS = 5
RECOMMENDATIONS = 5


class Rating(TypedDict):
    movie: str
    rating: int


class User(TypedDict):
    name: str
    ratings: list[Rating]


class ImdbDirector(TypedDict):
    id: str
    displayName: str


class ImdbImage(TypedDict):
    url: str


class ImdbMovie(TypedDict):
    id: str
    primaryTitle: str | None
    originalTitle: str | None
    primaryImage: ImdbImage | None
    directors: list[ImdbDirector]
    plot: str | None
    startYear: int | None


def load_users() -> list[User]:
    """Load data from the users.json file"""

    with open("data/users.json") as f:
        return cast(list[User], json.load(f))


def load_movies() -> list[ImdbMovie]:
    """Load data from the movies.json file"""

    with open("data/movies.json") as f:
        return cast(list[ImdbMovie], json.load(f))


class UserRating(TypedDict):
    """User rating for a movie"""

    user: str
    movie: str
    rating: int


def flatten_ratings(users: list[User]) -> list[UserRating]:
    """
    Flatten the user rating data into a list of UserRating objects

    This makes a CSV-like format for easier data manipulation and analysis.
    """

    return [
        UserRating(
            user=user["name"],
            movie=rating["movie"],
            rating=rating["rating"],
        )
        for user in users
        for rating in user["ratings"]
    ]


class Recommendation(TypedDict):
    movie_id: str
    avg_rating: float
    rating_count: int


def get_user_recommendations(
    user_name: str,
    ratings: pd.DataFrame,
    cluster_users: list[str],
    n: int,
    min_ratings: int,
) -> tuple[list[Recommendation], list[Recommendation]]:
    """
    Get movie recommendations and anti-recommendations for a specific user.

    Args:
        user_name: Name of the user to get recommendations for
        ratings: Flattened ratings dataframe with columns: user, movie, rating
        cluster_users: List of users in the same cluster as the target user
        n: Number of recommendations to return
        min_ratings: Minimum number of ratings required for a movie

    Returns:
        Tuple of (recommendations, anti_recommendations) where each is a list of
        Recommendation dictionaries
    """

    # All ratings from the target User's cluster
    cluster_ratings = ratings[ratings["user"].isin(cluster_users)]

    # Calculate average rating for each movie in the cluster
    movie_avg_ratings = (
        cluster_ratings.groupby("movie")["rating"].agg(["mean", "count"]).reset_index()
    )
    movie_avg_ratings.columns = ["movie", "avg_rating", "rating_count"]

    # Movies the User has not rated
    target_user_ratings = ratings[ratings["user"] == user_name]
    rated_movies = set(target_user_ratings["movie"].tolist())
    unrated_movies = movie_avg_ratings[
        ~movie_avg_ratings["movie"].isin(list(rated_movies))
    ]

    # Movies with at least min_ratings ratings
    reliable_movies = unrated_movies[unrated_movies["rating_count"] >= min_ratings]

    if len(reliable_movies) < n:
        print(f"Only {len(reliable_movies)} movies have {min_ratings}+ ratings")
        print(f"Including movies with fewer ratings to reach {n} recommendations")
        reliable_movies = unrated_movies

    if len(reliable_movies) == 0:
        print(f"No unrated movies found for User '{user_name}'")
        return [], []

    # Sort by average rating to get the top recommendations
    sorted_movies = reliable_movies.sort_values(by="avg_rating", ascending=False)
    top_recommendations = sorted_movies.head(n)
    top_anti_recommendations = sorted_movies.tail(n)

    recommendations = [
        Recommendation(
            movie_id=str(row["movie"]),
            avg_rating=float(row["avg_rating"]),
            rating_count=int(row["rating_count"]),
        )
        for _, row in top_recommendations.iterrows()
    ]

    anti_recommendations = [
        Recommendation(
            movie_id=str(row["movie"]),
            avg_rating=float(row["avg_rating"]),
            rating_count=int(row["rating_count"]),
        )
        for _, row in top_anti_recommendations.iterrows()
    ]

    return recommendations, anti_recommendations


def print_recommendations(
    recommendations: list[Recommendation],
    movies_by_id: dict[str, ImdbMovie],
) -> None:
    for rec in recommendations:
        movie_id = rec["movie_id"]
        movie_info = movies_by_id.get(movie_id, {})

        title = (
            movie_info.get("primaryTitle")
            or movie_info.get("originalTitle")
            or movie_id
        )
        year = movie_info.get("startYear", "N/A")
        print(f"\n{title} ({year})")
        print(
            f"  Cluster avg rating: {rec['avg_rating']:.2f}/10 ({rec['rating_count']} rating(s))"
        )

        plot = movie_info.get("plot")
        if plot:
            print(f"  Plot: {plot[:150]}{'...' if len(plot) > 150 else ''}")


def prompt_target_user(available_users: list[User]) -> str:
    options = [user["name"] for user in available_users]
    menu = TerminalMenu(options, title="Dla kogo przygotować rekomendacje?")
    selected_index = menu.show()
    if selected_index is None:
        raise ValueError("Didn't select a User")
    return options[selected_index]


def main() -> None:
    users = load_users()
    user = prompt_target_user(users)
    ratings = flatten_ratings(users)
    movies = load_movies()

    movies_by_id = {movie["id"]: movie for movie in movies}
    avg_rating = sum(rating["rating"] for rating in ratings) / len(ratings)

    data = pd.DataFrame(ratings)
    user_movie_matrix = data.pivot_table(index="user", columns="movie", values="rating")
    # Fill missing values with each User's average rating
    user_movie_matrix = user_movie_matrix.T.fillna(user_movie_matrix.mean(axis=1)).T

    print(f"Users: {user_movie_matrix.shape[0]}")
    print(f"Movies: {user_movie_matrix.shape[1]}")
    print(f"Ratings: {len(ratings)}")
    print(f"Average rating: {avg_rating:.2f}")
    print()

    kmeans = KMeans(n_clusters=CLUSTERS, random_state=123)
    cluster_labels = kmeans.fit_predict(user_movie_matrix)

    # Cluster the rating data
    user_movie_with_clusters = user_movie_matrix.copy()
    user_movie_with_clusters["cluster"] = cluster_labels
    if user not in user_movie_with_clusters.index:
        print(f"User '{user}' not found in the dataset!")
        return

    # Find the target User's cluster
    target_cluster = user_movie_with_clusters.loc[user, "cluster"]
    cluster_users = user_movie_with_clusters[
        user_movie_with_clusters["cluster"] == target_cluster
    ].index.tolist()
    print(f"Cluster: {target_cluster}")
    print(f"Users ({len(cluster_users)}): {', '.join(str(u) for u in cluster_users)}")

    # Compute recommendations for the target user
    recommendations, anti_recommendations = get_user_recommendations(
        user_name=user,
        ratings=data,
        cluster_users=cluster_users,
        n=RECOMMENDATIONS,
        min_ratings=2,
    )

    print(f"\n{'=' * 50}")
    print(f"Recommendations for {user}:")
    print(f"{'=' * 50}")
    print_recommendations(recommendations, movies_by_id)

    print(f"\n{'=' * 50}")
    print(f"Anti-recommendations for {user}:")
    print(f"{'=' * 50}")
    print_recommendations(anti_recommendations, movies_by_id)


if __name__ == "__main__":
    main()
