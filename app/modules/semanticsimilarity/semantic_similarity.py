import numpy as np
from gensim.models import Word2Vec
from gensim.models.fasttext import FastText
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sample_input import STOPWORDS


def vectorize_tfidf(sentences: list) -> np.ndarray:
    vectorizer = TfidfVectorizer(stop_words=STOPWORDS, ngram_range=(1, 2))
    sentences_matrix = vectorizer.fit_transform(sentences)

    return sentences_matrix.toarray()


def vectorize_word2vec(sentences: list) -> np.ndarray:
    tokenized_sentences = [sentence.lower().split() for sentence in sentences]
    vectorizer = Word2Vec(
        sentences=tokenized_sentences,
        vector_size=200,
        window=5,
        min_count=1,
        sg=1,
        epochs=5,
    )
    vectors = list()

    for sentence in tokenized_sentences:
        v = [vectorizer.wv[word] for word in sentence if word in vectorizer.wv]
        vectors.append(np.mean(v, axis=0) if v else np.zeros(vectorizer.vector_size))

    return np.array(vectors)


def vectorize_fasttext(sentences: list) -> np.ndarray:
    tokenized_sentences = [sentence.lower().split() for sentence in sentences]
    vectorizer = FastText(
        sentences=tokenized_sentences,
        vector_size=200,
        window=3,
        min_count=1,
        sg=1,
        epochs=5,
    )
    vectors = list()

    for sentence in tokenized_sentences:
        v = [vectorizer.wv[word] for word in sentence if word in vectorizer.wv]
        vectors.append(np.mean(v, axis=0) if v else np.zeros(vectorizer.vector_size))

    return np.array(vectors)


def svd_reduction(matrix, components) -> np.ndarray:
    svd = TruncatedSVD(
        n_components=min(components, min(matrix.shape) - 1), random_state=1405
    )

    return svd.fit_transform(matrix)


def similarity(matrix):
    return cosine_similarity(matrix)


METHODS = {
    "tfidf": vectorize_tfidf,
    "word2vec": vectorize_word2vec,
    "fasttext": vectorize_fasttext,
}


def pipeline(
    sentences: list,
    method: str,
    components: int | None = None,
    round_decimals: int = 3,
    return_reduced: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:

    if method not in METHODS:
        raise ValueError("Undefined Method!")
    if return_reduced and components is None:
        raise ValueError("Cannot apply SVD reducion without setting components!")
    matrix = METHODS[method](sentences)

    if components is not None:
        matrix = svd_reduction(matrix, components)
        reduced = matrix
    sim = np.round(similarity(matrix), round_decimals)

    if return_reduced:
        return (sim, reduced)

    return sim