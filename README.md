# IndoorSceneRecognition
"""
The model is used for Indoor scene recoginition. The repo consists of 2 files: utils, ModelBuildAndPrediction. 
utils: Contains all the predefined functions used to build the model. Most of the imports are done in utils. 
ModelBuildAndPrediction: Contains the neural network CNN with just one hidden layer to train the data after restricting the mimages to 100 pixels each. The model runs better with a GPU. Also, use this section to pass the testing image to retrieve the end result or the prediction for your test image.

Details: 1 hidden layer. Epoch is set to 5 to reduce run time.The images in the input dataset also determines the run time for each Epoch.

The Dataset needed can be fetched using the link below: 
http://web.mit.edu/torralba/www/indoor.html
"""
