from flask import Flask, jsonify, request

from project_6_content_deply import contentBasedRec

app = Flask(__name__)

recommender = contentBasedRec(df_csv="df_reduced.csv")


@app.route("/")
def index():
    return """
<h2>Welcome to StormWhale Content Based recommender system</h2>

<p>To get a list of user IDs, use this format:</p>
<code>/get_user_list?top_n=XX </code>
<p><small>(top_n is optional and defaults to 5)</small></p>

<p>To get a list of movies, use this format:</p>
<code>/get_movie_list?top_n=XX</code>
<p><small>(top_n is optional and defaults to 5)</small></p>

<p>For content-based recommendations, use this format:</p>
<code>/recommend?user_id=XXX&top_n=XXX</code>

<p class="examples">Some user_id examples: 394716, 173270, 168663, 3785</p>

"""


@app.route("/recommend", methods=["GET"])
def recommend():
    user_id = request.args.get("user_id", type=int)
    top_n = request.args.get("top_n", default=5, type=int)

    if user_id is None:
        return jsonify(
            {
                "error": "Missing user_id parameter\nEnter with this format:/recommend?user_id=XXX&top_n=XXX"
            }
        ), 400
    try:
        rec_list = recommender.content_based_recommendations(
            user_id=user_id, top_n=top_n
        )
        recs = rec_list[["movieId", "title", "features", "cb_score"]].to_dict(
            orient="records"
        )

        # Build simple HTML output
        html = f"<h2>Recommendations for User {user_id}</h2>"
        if not recs:
            html += "<p>No recommendations found.</p>"
        else:
            html += "<table border='1' cellpadding='5' cellspacing='0'>"
            html += "<tr><th>Movie ID</th><th>Title</th><th>Genres & Features</th><th>Score</th></tr>"
            for r in recs:
                html += f"<tr><td>{r['movieId']}</td><td>{r['title']}</td><td>{r['features']}</td><td>{r['cb_score']:.3f}</td></tr>"
            html += "</table>"

        return html

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/get_user_list", methods=["GET"])
def get_user_list():
    top_n = request.args.get("top_n", default=5, type=int)
    users = recommender.get_user_list(top_n=top_n)
    return jsonify({"users": users})


@app.route("/get_movie_list", methods=["GET"])
def get_movie_list():
    top_n = request.args.get("top_n", default=5, type=int)
    movies = recommender.get_movie_list(
        top_n=top_n
    )  # list of dicts with 'title' and 'genres'

    rows = ""
    for movie in movies:
        rows += f"<tr><td>{movie['title']}</td><td>{movie['genres']}</td></tr>"

    html = f"""
    <html>
    <head>
        <title>Movie List</title>
        <style>
            table {{ border-collapse: collapse; width: 60%; margin: 20px auto; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; }}
            th {{ background-color: #f2f2f2; }}
            tr:hover {{ background-color: #f9f9f9; }}
            body {{ font-family: Arial, sans-serif; }}
            h2 {{ text-align: center; }}
        </style>
    </head>
    <body>
        <h2>A List of {top_n} Movies</h2>
        <table>
            <thead>
                <tr>
                    <th>Title</th>
                    <th>Genres</th>
                </tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>
    </body>
    </html>
    """
    return html


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
