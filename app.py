import streamlit as st
from src import svd_compression as svc, linear_system as ls, file_handler as fp


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

page = st.session_state.page


if page == "pg1":

    st.title("SVD Image Compression")

    # File upload
    uploaded_file = st.file_uploader(
        "Upload a file",
        key="page1_upload",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"]
    )

    # Dropdown
    option = st.selectbox(
    "Mode",
    ["Components", "Percentage"],
    accept_new_options=False
    )

    # Slider
    slider_value = st.slider(
        "Slider",
        min_value=1,
        max_value=100,
        value=50
    )

    if uploaded_file is not None:

        st.divider()
        st.subheader("Compressed Image")

        image = svc.compress(fp.image_to_array(uploaded_file), slider_value)

        st.image(
            image,
            caption="Compressed Image",
            use_container_width=False
        )
elif page == "pg2":

    st.title("Least Squares vs. Linear Regression")

    # CSV Upload
    uploaded_csv = st.file_uploader(
        "Upload CSV",
        type=["csv"],
        key="page2_upload"
    )

    if uploaded_csv is not None:
        data = fp.csv_to_array(uploaded_csv)
        x, residual, rank, s = ls.least_squares(data)
        reg = ls.regression(data)

        st.latex(r"CSV\ Contents = " + fp.array_to_bmatrix(data))

        st.divider()

        st.header("Output")

        st.subheader("NumPy Least Squares")

        Output = fr"""
        x = {fp.array_to_bmatrix(x)}\qquad\qquad
        Ax = {fp.array_to_bmatrix(data[:, :-1] @ x)}\qquad\qquad
        b = {fp.array_to_bmatrix(data[:, -1])}
        """

        st.latex(Output)

        st.subheader("Scikit-Learn Linear Regression")

        Output = fr"""
        x = {fp.array_to_bmatrix(reg.coef_)}\qquad\qquad
        Ax = {fp.array_to_bmatrix(data[:, :-1] @ reg.coef_)}\qquad\qquad
        b = {fp.array_to_bmatrix(data[:, -1])}
        """

        st.latex(Output)
    else:
        st.info("Upload a CSV file to view its contents.")

