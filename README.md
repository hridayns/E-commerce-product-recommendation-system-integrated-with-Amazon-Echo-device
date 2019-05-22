# Recommendation System for Amazon Alexa E-Commerce Application

- Researched, planned and developed a personalized product recommendation engine from scratch, to be deployed as a micro service for ecommerce shopping cart applications.
- Did detailed research, including studying research papers and evaluated types of recommender systems.
- Trained, tested and developed a production ready recommender system using Tensorflow, sklearn, numPy, pandas, sciPy, Flask, Flask-PyMongo and MongoDB (NoSQL). It was built on Cosine similarities between TF-IDF vectors in vector space representation algorithm for content-based filtering combined with Matrix Factorization model using WALS algorithm to optimize the loss function for collaborative filtering.
- Collaborated with the apigee and Amazon Alexa teams to integrate the output of the recommender system to an Amazon Echo device.

This repository consists of a Recommendation engine using the Collaborative filtering and Content Based filtering methods in TensorFlow.

**Dataset used:** Amazon ecommerce sales dataset of ~10000 training examples, with ~15 features each was split for use in training and testing of the algorithms.

1. *collab .py* contains code for the collaborative filtering aspect of the recommendation engine.
2. *contentTensor .py* contains code for the content-based aproach of the recommendation engine.
3. *flaskTest1 .py* contains code for exposing both content-based and collaborative filtering methods as an API that returns JSON output using Python Flask.