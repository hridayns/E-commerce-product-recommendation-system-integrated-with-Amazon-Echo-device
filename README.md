# Recommendation System for Amazon Alexa E-Commerce Application

This repository consists of a Recommendation engine using the Collaborative filtering and Content Based filtering methods in TensorFlow.

**Dataset used:** Amazon ecommerce sales dataset of ~10000 training examples, with ~15 features each was split for use in training and testing of the algorithms.

1. *collab .py* contains code for the collaborative filtering aspect of the recommendation engine.
2. *contentTensor .py* contains code for the content-based aproach of the recommendation engine.
3. *flaskTest1 .py* contains code for exposing both content-based and collaborative filtering methods as an API that returns JSON output using Python Flask.