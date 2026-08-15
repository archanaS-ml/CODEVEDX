from flask import Flask, request, jsonify, render_template
import recommender

app = Flask(__name__)

recommender.create_sample_data()
catalog_df = recommender.load_data()
vectorizer, item_vectors = recommender.build_content_profiles(catalog_df)


@app.route("/")
def home():
    titles = sorted(catalog_df["title"].tolist())
    catalog = catalog_df.sort_values("title").to_dict(orient="records")
    return render_template("index.html", titles=titles, catalog=catalog)


@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        data = request.get_json(force=True)
        mode = data.get("mode")
        query = (data.get("query") or "").strip()

        if not query:
            return jsonify({"error": "Please enter a course title or a preference."}), 400

        if mode == "title":
            results = recommender.recommend_by_title(query, catalog_df, item_vectors)
            if results is None:
                return jsonify({"error": f"'{query}' was not found in the catalog."}), 404
        elif mode == "text":
            results = recommender.recommend_by_preferences(
                query, catalog_df, vectorizer, item_vectors
            )
        else:
            return jsonify({"error": "Invalid recommendation mode."}), 400

        if not results:
            return jsonify({"error": "No close matches found. Try different keywords."}), 200

        return jsonify({"results": results})

    except Exception as e:
        return jsonify({"error": f"Something went wrong: {e}"}), 500


if __name__ == "__main__":
    app.run(debug=True)