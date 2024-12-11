import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from PyQt5.QtWidgets import QApplication, QFileDialog, QMainWindow
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Tokenize text
    words = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    # Stemming
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    
    return words
class DocumentStorage:
    def __init__(self):
        self.document_data = {}

    def add_document(self, doc_id, processed_text):
        # Store processed text in hash table
        self.document_data[doc_id] = processed_text

    def get_all_documents(self):
        return self.document_data
def compute_prefix_function(pattern):
    prefix = [0] * len(pattern)
    j = 0
    for i in range(1, len(pattern)):
        while j > 0 and pattern[i] != pattern[j]:
            j = prefix[j - 1]
        if pattern[i] == pattern[j]:
            j += 1
        prefix[i] = j
    return prefix

def kmp_search(text, pattern):
    prefix = compute_prefix_function(pattern)
    j = 0
    matches = []
    for i in range(len(text)):
        while j > 0 and text[i] != pattern[j]:
            j = prefix[j - 1]
        if text[i] == pattern[j]:
            j += 1
        if j == len(pattern):
            matches.append(i - j + 1)
            j = prefix[j - 1]
    return matches
def calculate_cosine_similarity(doc1, doc2):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([doc1, doc2])
    return cosine_similarity(vectors[0:1], vectors[1:])[0][0]
def generate_shingles(text, k=3):
    # Generate k-grams
    shingles = set()
    for i in range(len(text) - k + 1):
        shingles.add(text[i:i + k])
    return shingles
def cluster_documents(document_vectors, num_clusters=5):
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(document_vectors)
    return kmeans.labels_
class PlagiarismApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Plagiarism Detection System")
        self.setGeometry(200, 200, 600, 400)

    def upload_document(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Upload Document", "", "Text Files (*.txt)")
        if file_path:
            with open(file_path, 'r') as file:
                document_text = file.read()
                return document_text

