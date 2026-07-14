import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report
from sklearn.metrics import f1_score


df = pd.read_csv("data/products.csv")
df.fillna("Unknown", inplace=True)

df["category"] = df["category"].replace({

    # Kurta Family
    "Kurti": "Kurta",
    "Straight Kurta": "Kurta",

    # Pants Family
    "Cargo Pants": "Pants",
    "Track Pants": "Pants",
    "Palazzo": "Pants",
    "Trousers": "Pants",
    "Jeans": "Pants",

    # Tops
    "Hoodie": "Top",
    "Sweatshirt": "Top",
    "T-Shirt": "Top",
    "Tunic": "Top",

    # Outerwear
    "Blazer": "Jacket"
})



df["color"] = df["color"].replace({

    # Blues
    "Navy Blue": "Blue",
    "Sky Blue": "Blue",
    "Dark Blue": "Blue",
    "Turquoise": "Blue",

    # Greens
    "Olive Green": "Green",
    "Emerald Green": "Green",
    "Dark Green": "Green",
    "Mint Green": "Green",

    # Reds
    "Maroon": "Red",
    "Rust Orange": "Orange",

    # Pinks
    "Coral": "Pink",
    "Peach": "Pink",

    # Greys
    "Charcoal Grey": "Grey",

    # Whites
    "Cream": "White",
    "Ivory": "White",

    # Gold/Silver
    "Golden": "Yellow"
})



df["fabric"] = df["fabric"].replace({

    "Georgette": "Synthetic",
    "Chiffon": "Synthetic",
    "Polyester": "Synthetic",

    "Velvet": "Silk",
    "Satin": "Silk",

    "Lace": "Cotton"
})

df["neckline"] = df["neckline"].replace({

    "Crew Neck": "Round Neck",

    "Mandarin Collar": "Collar",

    "Hood": "Collar",

    "Square Neck": "Round Neck",

    "Sweetheart Neck": "V-Neck"
})


df["sleeve"] = df["sleeve"].replace({

    "Short Sleeve": "Half Sleeve",

    "3/4 Sleeve": "Half Sleeve",

    "Long Sleeve": "Full Sleeve"
})


df["length"] = df["length"].replace({

    "Knee Length": "Midi"
})


df["silhouette"] = df["silhouette"].replace({

    "Regular Fit": "Straight",

    "Slim Fit": "Straight",

    "Relaxed Fit": "Straight",

    "Wide Leg": "Straight",

    "Bodycon": "Fit & Flare",

    "Anarkali": "A-Line"
})


df["embellishment"] = df["embellishment"].replace({

    "Checked": "Printed",

    "Striped": "Printed",

    "Floral": "Printed",

    "Lace": "Solid",

    "Zari": "Embroidered"
})


X = df["description"]
vectorizer = TfidfVectorizer(lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=1)
X_vectorized = vectorizer.fit_transform(X)

attributes = [
    "category",
    "color",
    "fabric",
    "neckline",
    "sleeve",
    "length",
    "silhouette",
    "embellishment"
]

models = {}
for attribute in attributes:

    

    print(f"\nTraining model for: {attribute}")

    y = df[attribute]

    X_train, X_test, y_train, y_test = train_test_split(
        X_vectorized,
        y,
        test_size=0.2,
        random_state=42
    )

    model = LogisticRegression(max_iter=1000,class_weight="balanced")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"{attribute} Accuracy: {accuracy:.2f}")
    print(f"{attribute} F1 Score: {f1:.2f}")
    print(f"\n{attribute}---------------")
    print(classification_report(y_test, y_pred))


    models[attribute] = model




joblib.dump(vectorizer, "models/vectorizer.pkl")

print("Vectorizer Saved")


for attribute, model in models.items():

    joblib.dump(
        model,
        f"models/{attribute}_model.pkl"
    )

    print(f"{attribute} model saved.")

