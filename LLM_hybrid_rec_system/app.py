from flask import Flask, request, jsonify
from project_6_deply import HybridRecommender
from project_6_deply import prepare_dataset

app = Flask(__name__)

prepare_dataset("df_reduce.csv")

recommender = HybridRecommender(df_csv="df_reduce.csv")


@app.route("/")
def index():
    return """
<h2>Welcome to StormWhale hybrid recommender system</h2>
<p>
For hybrid recommendations use this format: <br>
<code>/recommend?user_id=XXX&top_n=XXX&alpha=XXX'</code><br><br>
For Gemma 3 explanations, enter with this format:<br>
<code>/explain?user_id=XXX&top_n=XXX&alpha=XXX'</code><br><br>
Some user_id examples: 346368, 173270, 168663<br>
top_n is for top number recommendations<br>
Alpha is the hybrid factor (0-1). Higher for more content-based filtering.<br>
Alpha is default at 0.5.
"""


@app.route("/recommend", methods=["GET"])
def recommend():
    user_id = request.args.get("user_id", type=int)
    top_n = request.args.get("top_n", default=5, type=int)
    alpha = request.args.get("alpha", default=0.5, type=float)

    if user_id is None:
        return jsonify(
            {
                "error": "Missing user_id parameter\nEnter with this format:/recommend?user_id=XXX&top_n=XXX&alpha=XXX"
            }
        ), 400
    try:
        rec_list = recommender.hybrid_recommender_for_user(
            user_id=user_id, top_n=top_n, alpha=alpha
        )
        recs = rec_list[["movieId", "title", "genres", "hybrid_score"]].to_dict(
            orient="records"
        )
        return jsonify({"user_id": user_id, "recommendations": recs})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/explain", methods=["GET"])
def explain():
    user_id = request.args.get("user_id", type=int)
    top_n = request.args.get("top_n", default=5, type=int)
    alpha = request.args.get("alpha", default=0.5, type=float)

    if user_id is None:
        return jsonify(
            {
                "error": "Missing user_id parameter\nEnter with this format:/explain?user_id=XXX&top_n=XXX&alpha=XXX"
            }
        ), 400

    try:
        explanation = recommender.llm_explain(
            user_id=user_id,
            top_n=top_n,
            alpha=alpha,
            model="gemma3:latest",
            include_content=True,
        )
        return jsonify({"user_id": user_id, "explanation": explanation})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
