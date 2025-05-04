from setuptools import setup, find_packages

setup(
    name="lal",
    version="0.1.0",
    description="Human-like Thinking LLM System",
    author="",
    author_email="",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "tqdm",
        "numpy",
        "accelerate",
    ],
    python_requires=">=3.8",
)
