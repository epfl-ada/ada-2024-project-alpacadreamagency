import pandas as pd

def save_predictions(predictions, testing_set, genre_labels):
    predicted_genre = []

    print("Preparing output...")
    for movie_prediction in predictions:
        genres = [genre_labels[i] for i, is_genre in enumerate(movie_prediction) if is_genre == 1]
        predicted_genre.append(genres)

    predictins_output = pd.DataFrame({
        'wikipedia_movie_ID': testing_set["wikipedia_movie_ID"],
        'name': testing_set["name"],
        'original_genres': testing_set["new_genres"],
        'predicted_genres': predicted_genre,
    })

    print("Writing in file...")
    predictins_output.to_csv("movies_predicted_genre.csv", index = False)

    print("DONE!")