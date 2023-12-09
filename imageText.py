# import pandas as pd
# import tensorflow as tf
# from scipy.sparse import csr_matrix
# from tensorflow.keras.callbacks import ModelCheckpoint
# from sklearn.feature_extraction.text import TfidfVectorizer
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, Dense
# from sklearn.metrics.pairwise import cosine_similarity
# from keras.models import load_model

# class ImageText:
#     def __init__(self):
#         self.img_size = (512, 512)
#         self.model = load_model('D:\hackathon\FASTAPI\medicine.h5')
#         self.csv_file_path = 'D:\hackathon\FASTAPI\medicine.csv'
#         self.df = pd.read_csv(self.csv_file_path)
#         self.df['combined_features'] = self.df['medicine_desc'] + ' ' + self.df['salt_composition']

#         # Create tfidf_vectorizer and tfidf_matrix here
#         self.tfidf_vectorizer = TfidfVectorizer()
#         self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.df['combined_features'])

#         input_layer = Input(shape=(self.tfidf_matrix.shape[1],))
#         embedding_layer = Dense(128, activation='relu')(input_layer)
#         output_layer = Dense(128)(embedding_layer)

#         self.model = Model(inputs=input_layer, outputs=output_layer)
#         self.model.compile(optimizer='adam', loss='mean_squared_error')

#     def find_similar_medicines(self, query_name, top_n=10, similarity_threshold=0.9):
#         query_features = self.df.loc[self.df['product_name'] == query_name, 'combined_features'].values
#         if len(query_features) == 0:
#             return "Medicine not found in the dataset."

#         query_vector = self.model.predict(self.tfidf_vectorizer.transform(query_features))

#         similarity_scores = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
#         similarity_scores[self.df['product_name'] == query_name] = 0.0

#         similar_indices = [i for i, score in enumerate(similarity_scores) if score >= similarity_threshold]

#         similar_medicines = self.df.iloc[similar_indices].sort_values(by='product_name', ascending=True).head(top_n)

#         return similarity_scores, similar_medicines

# # Instantiate the class
# image_text_instance = ImageText()

# # Example usage
# query_medicine = "Human Insulatard 40IU/ml Suspension for Injection"
# similarity_scores, similar_medicines = image_text_instance.find_similar_medicines(query_medicine)

# print(f"Similarity Scores: {similarity_scores}")
# print("\nTop Similar Medicines:")
# for idx, row in similar_medicines.iterrows():
#     similarity_percentage = similarity_scores[image_text_instance.df['product_name'] == row['product_name']][0] * 100
#     print(f"{row['product_name']} - Similarity: {similarity_percentage:.2f}%")
#     print(f"   Description: {row['medicine_desc']}")
#     print(f"   Composition: {row['salt_composition']}")
#     print(f"   price: {row['product_price']}\n")
