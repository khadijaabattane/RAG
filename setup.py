from setuptools import setup, find_packages

setup(
    name='rag_pdf_MAIF',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        "streamlit",
        "transformers",
        "torch",
        "PyPDF2",
        "scikit-learn",
        "faiss-cpu",
        "langchain-openai",
        "langchain-text-splitters",
        "tqdm"
    ],
    author='Khadija ABATTANE',
    include_package_data=True,
)
