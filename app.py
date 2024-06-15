import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def read_files(file_dir):
    student_files = [doc for doc in os.listdir(file_dir) if doc.endswith('.txt')]
    student_notes = [open(os.path.join(file_dir, _file), encoding='utf-8').read()
                     for _file in student_files]
    return student_files, student_notes

def vectorize(text_data):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(text_data)
    return vectors, vectorizer

def similarity(doc1, doc2, vectorizer):
    tfidf_doc1 = vectorizer.transform([doc1])
    tfidf_doc2 = vectorizer.transform([doc2])
    return cosine_similarity(tfidf_doc1, tfidf_doc2)[0][0]

def check_plagiarism(vectors, vectorizer, student_files):
    plagiarism_results = set()
    for i, text_vector_a in enumerate(vectors):
        for text_vector_b in vectors[:i]:
            sim_score = similarity(text_vector_a.toarray()[0], text_vector_b.toarray()[0], vectorizer)
            student_pair = (student_files[i], student_files[vectors.tolist().index(text_vector_b)])
            plagiarism_results.add((student_pair[0], student_pair[1], sim_score))
    return plagiarism_results

file_dir = '.'
student_files, student_notes = read_files(file_dir)
vectors, vectorizer = vectorize(student_notes)
plagiarism_results = check_plagiarism(vectors, vectorizer, student_files)

for data in plagiarism_results:
    print(data)