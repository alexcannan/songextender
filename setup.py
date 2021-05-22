from setuptools import setup, find_namespace_packages

setup(
    name='songextender',
    version='1',
    python_requires='>=3.8',
    author='Alex Cannan',
    author_email='alexfcannan@gmail.com',
    packages=find_namespace_packages(include=['songextender.*']),
    long_description="extend songs using openai jukebox",
    install_requires=[
        "ipython",
    ]
)

