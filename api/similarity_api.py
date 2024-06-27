from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)

# Load a pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

@app.route('/cosine_similarity', methods=['POST'])
def cosine_similarity():
    data = request.get_json()

    if 'sentence1' not in data or 'sentence2' not in data:
        return jsonify({'error': 'Please provide both sentence1 and sentence2 in the request body'}), 400

    sentence1 = data['sentence1']
    sentence2 = data['sentence2']

    # Encode the sentences using the pre-trained model
    embedding1 = model.encode(sentence1, convert_to_tensor=True)
    embedding2 = model.encode(sentence2, convert_to_tensor=True)

    # Compute cosine similarity between the two sentence embeddings
    cosine_sim = util.cos_sim(embedding1, embedding2).item()

    return jsonify({'cosine_similarity': cosine_sim})

if __name__ == '__main__':
    app.run(debug=True)
