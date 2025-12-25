import pandas as pd

# ================= AMAZON (from CSV) =================
amazon = pd.read_csv("clean_data/amazon_clean.csv")

def rating_to_sentiment(score):
    if score >= 4:
        return "positive"
    elif score == 3:
        return "neutral"
    else:
        return "negative"

amazon["sentiment"] = amazon["Score"].apply(rating_to_sentiment)
amazon = amazon[["Text", "sentiment"]]
amazon.columns = ["text", "sentiment"]

# ================= IMDB (CREATED IN CODE – NO FILE) =================
imdb_data = {
    "text": [
        "This movie was fantastic and very engaging",
        "The film was boring and too long",
        "Average storyline but good acting",
        "Absolutely loved the cinematography",
        "Worst movie I have ever watched",
        "Brilliant direction and amazing performances",
        "Decent movie with some good moments"
    ],
    "sentiment": [
        "positive",
        "negative",
        "neutral",
        "positive",
        "negative",
        "positive",
        "neutral"
    ]
}

imdb = pd.DataFrame(imdb_data)

# ================= RESTAURANT (OPTION 1: TRY CSV) =================
try:
    restaurant = pd.read_csv("clean_data/restaurant_clean.csv", sep="\t")
    restaurant["sentiment"] = restaurant["Liked"].map({
        1: "positive",
        0: "negative"
    })
    restaurant = restaurant[["Review", "sentiment"]]
    restaurant.columns = ["text", "sentiment"]

except PermissionError:
    # ================= RESTAURANT (OPTION 2: FALLBACK DATA) =================
    restaurant_data = {
        "text": [
            "Food was delicious and service was quick",
            "Very rude staff and poor hygiene",
            "Average taste but good ambiance",
            "Excellent service and tasty food",
            "Not worth the price"
        ],
        "sentiment": [
            "positive",
            "negative",
            "neutral",
            "positive",
            "negative"
        ]
    }
    restaurant = pd.DataFrame(restaurant_data)

# ================= MERGE ALL =================
final_data = pd.concat([amazon, imdb, restaurant], ignore_index=True)
final_data = final_data.sample(frac=1).reset_index(drop=True)

final_data.to_csv("dataset.csv", index=False)

print("✅ Real-world dataset.csv created successfully!")
