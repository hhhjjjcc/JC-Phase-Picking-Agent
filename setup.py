from setuptools import setup, find_packages

setup(
    name="seismic_agent",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "obspy",
        "tensorflow",
        "flask",
        "flask-socketio",
        "plotly",
        "pandas",
        "scipy",
        "matplotlib",
        "tqdm",
        "h5py"
    ],
    python_requires=">=3.6",
) 