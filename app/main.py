import streamlit as st
from modules import linear_system as ls
from modules import svd_compression as svc
from modules.pagerank import page_rank as pg
from modules.pagerank import sample_input as si
from modules.semanticsimilarity.semantic_similarity import pipeline
from utils import file_handler as fp
from utils.graph_visualizer import plot_digraph

st.set_page_config(page_title="linalg", layout="wide")

# Initialize page
if "page" not in st.session_state:
    st.session_state.page = "pg1"

# Sidebar
with st.sidebar:
    st.title("Navigation")

    if st.button("SVD Image Compression", use_container_width=True):
        st.session_state.page = "pg1"

    if st.button("Least Squares vs. Linear Regression", use_container_width=True):
        st.session_state.page = "pg2"

    if st.button(label="PageRank", use_container_width=True):
        st.session_state.page = "pg3"

    if st.button(label="Semantic Similarity", use_container_width=True):
        st.session_state.page = "pg4"

page = st.session_state.page


if page == "pg1":
    st.title("SVD Image Compression")

    # File upload
    uploaded_file = st.file_uploader(
        "Upload a file",
        key="page1_upload",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
    )

    # Dropdown
    option = st.selectbox(
        "Mode", ["Components", "Percentage"], accept_new_options=False
    )

    # Slider
    slider_value = st.slider("Slider", min_value=1, max_value=100, value=50)

    if uploaded_file is not None:
        st.divider()
        st.subheader("Compressed Image")

        image = svc.compress(
            fp.image_to_array(uploaded_file),
            slider_value,
            True if option == "Percentage" else False,
        )

        st.image(image, caption="Compressed Image", use_container_width=False)
elif page == "pg2":
    st.title("Least Squares vs. Linear Regression")

    # CSV Upload
    uploaded_csv = st.file_uploader("Upload CSV", type=["csv"], key="page2_upload")

    if uploaded_csv is not None:
        data = fp.csv_to_array(uploaded_csv)
        x, residual, rank, s = ls.least_squares(data)
        reg = ls.regression(data)

        st.latex(r"CSV\ Contents = " + fp.array_to_bmatrix(data))

        st.divider()

        st.header("Output")

        st.subheader("NumPy Least Squares")

        Output = rf"""
        x = {fp.array_to_bmatrix(x)}\qquad\qquad
        Ax = {fp.array_to_bmatrix(data[:, :-1] @ x)}\qquad\qquad
        b = {fp.array_to_bmatrix(data[:, -1])}
        """

        st.latex(Output)

        st.subheader("Scikit-Learn Linear Regression")

        Output = rf"""
        x = {fp.array_to_bmatrix(reg.coef_)}\qquad\qquad
        Ax = {fp.array_to_bmatrix(data[:, :-1] @ reg.coef_)}\qquad\qquad
        b = {fp.array_to_bmatrix(data[:, -1])}
        """

        st.latex(Output)
    else:
        st.info("Upload a CSV file to view its contents.")
elif page == "pg3":
    st.title("PageRank Implementation")

    st.subheader("Sample Graph")

    st.pyplot(plot_digraph(si.GRAPH), use_container_width=False)

    st.divider()

    Output = str()

    for item in pg.page_rank(si.GRAPH, si.DAMPING_FACTOR, si.TOLERANCE).items():
        Output += f"Node {item[0]} rank: {item[1]}\n"

    st.text(Output)

    st.divider()

    st.subheader("Code:")

    with open(r"app/modules/pagerank/page_rank.py", "r") as file:
        st.code(file.read(), language="python")
else:
    st.title("Semantic Similarity/SVD Reduction")

    st.set_page_config(layout="wide")

    # -------------------------
    # Sentence input
    # -------------------------
    sentences = st.text_area(
        "Sentences (One per line)",
        placeholder="Type here...",
        height=350,
    )

    st.write("")  # Spacer

    # -------------------------
    # Bottom row
    # -------------------------
    col1, col2, col3, col4 = st.columns([2.2, 1.6, 1.8, 1])

    with col1:
        method = st.selectbox(
            "Method",
            [
                "tfidf",
                "word2vec",
                "fasttext",
            ],
            index=0,
        )

    with col2:
        components = st.number_input(
            "Components",
            min_value=1,
            value=None,
            step=1,
        )

    with col3:
        round_decimals = st.number_input(
            "Round decimals",
            min_value=1,
            value=3,
            step=1,
        )

    with col4:
        st.write("")  # Vertical spacing
        st.write("")
        return_reduced = st.checkbox("Return Reduced")

    st.divider()
    if sentences != "":
        sentences = sentences.split("\n")

        pipe_out = pipeline(
            sentences=sentences,
            method=method,
            components=components,
            round_decimals=round_decimals,
            return_reduced=return_reduced,
        )

        if return_reduced:
            sim = pipe_out[0]
            reduced = pipe_out[1]

            Output = rf"""
            Similarity Matrix = {fp.array_to_bmatrix(sim)}\newline \newline
            SVD Reduced Matrix = {fp.array_to_bmatrix(reduced)}        
            """
        else:
            Output = f"""
            Similarity Matrix = {fp.array_to_bmatrix(pipe_out)}
            """
    else:
        Output = """"""

    st.latex(Output)
