from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity
from urllib.parse import unquote
import pandas as pd

app = FastAPI()

df = pd.read_csv('D:\hackathon\FASTAPI\medicine.csv')

df['combined_features'] = df['medicine_desc'] + ' ' + df['salt_composition']

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined_features'])

input_layer = Input(shape=(tfidf_matrix.shape[1],))
embedding_layer = Dense(128, activation='relu')(input_layer)
output_layer = Dense(128)(embedding_layer)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer='adam', loss='mean_squared_error')

class QueryModel(BaseModel):
    query_medicine: str
    top_n: int = 10
    similarity_threshold: float = 0.9

def find_similar_medicines(query_name, top_n=10, similarity_threshold=0.9):
    query_features = df.loc[df['product_name'] == query_name, 'combined_features'].values
    if len(query_features) == 0:
        return "Medicine not found in the dataset."

    query_vector = tfidf_vectorizer.transform(query_features)
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()

    similarity_scores[df['product_name'] == query_name] = 0.0

    similar_indices = [i for i, score in enumerate(similarity_scores) if score >= similarity_threshold]
    similar_medicines = df.iloc[similar_indices].sort_values(by='product_name', ascending=True).head(top_n)

    return similarity_scores, similar_medicines

# Define FastAPI endpoint for finding similar medicines
@app.post("/similar_medicines/")
async def get_similar_medicines(
    query_medicine: str = Form(..., title="Query Medicine", description="Enter the name of the medicine to find similar ones."),
    top_n: int = Form(10, title="Top N", description="Number of top similar medicines to retrieve."),
    similarity_threshold: float = Form(0.9, title="Similarity Threshold", description="Threshold for considering medicines as similar.")
):
    decoded_query_medicine = unquote(query_medicine)
    similarity_scores, similar_medicines = find_similar_medicines(decoded_query_medicine, top_n, similarity_threshold)

    if isinstance(similar_medicines, str):
        return JSONResponse(content={"error": similar_medicines}, status_code=404)

    response_data = {
        "query_medicine": decoded_query_medicine,
        "similarity_scores": similarity_scores.tolist(),
        "similar_medicines": []
    }

    for idx, row in similar_medicines.iterrows():
        similarity_percentage = similarity_scores[df['product_name'] == row['product_name']][0] * 100
        medicine_info = {
            "product_name": row['product_name'],
            "similarity_percentage": round(similarity_percentage, 2),
            "description": row['medicine_desc'],
            "composition": row['salt_composition'],
            "price": row['product_price']
        }
        response_data["similar_medicines"].append(medicine_info)

    return response_data

# Run the FastAPI application using Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
