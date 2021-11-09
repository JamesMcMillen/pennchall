# data-science-challenge

Welcome! The following challenge is to evaluate your strengths as a data scientist. The expectation is not that you would finish all of it, but represents all of the sorts of problems you might expect working on a typical problem. You should limit the exercise to no more than a few hours. When you're finished put your code in a VCS of your choice and share the link!

## Problem Statement

We are creating a machine learning algorithm that will predict the price of Bitcoin in USD in the next second based on the price from the last 60 seconds. The variable `price_high` is the target variable we wish to predict, our *Y*. The rest of the data can be used for your feature set, *X*. I have created a lookback function to expand the feature set and a min-max scaler to standardize the features. You can use this as your feature set, or modify it as you wish. The dataset consists of the first 10,000 seconds of Bitcoin data a single day, January 1, 2020 (until about 8:00 am). The data can be found in `bitcoin.csv`. We have a model in place, trained in `bitcoin-predictor.ipynb`. I started by using a deep neural network, because everyone keeps talking about how cool they are. Problem is, the model is not very good. Actually the model is completely awful and predicts `0` for everything. So, a lot of room for improvement!

The main challenge is this: can you make a model that actually works? You can use a neural network in tensorflow as I've done, or you can scrap it altogether and use regression, random forest, svm, anything you like. No model or package is off the table. Feel free to modify the jupyter notebook directly or put it in whatever workflow suits you best.

The main idea is to get a decent working model up and running quickly, but some other considerations you should think about are listed below. You don't have to answer all of the questions, but pick at least one or two and try to answer them (you can add your answers in this readme if you'd like).

* Does this data set even make sense? What are the limitations of this data set?
* 	Possible Limitations:
*		> Does not span a long enough time period
*		> Data has two main inflection points > there is spike in price up and then a spike in price down > this will cause a model not to converage > a larger time window of the data would help
*		> The main problem with this data is that it is per second, and as you can see from the chart, the price changes up and down from second to second
*		> Smoothing the data, and developing a model on the smoothed target might be a better option
*			> An example of this is in the code, where I created the mean of the rolling three seconds > compare this to the orginal data and the flux between seconds is not as substantial
*			> Unless there is a specific need to estimate the model per second, I would develop the model based on a rolling average of at least three seconds
* Is the lookback window of 60 seconds helpful? What are its limitations? What other features would you want to see in this data set?
*		> The lookback window is exactly what is needed to create a time-series dataset for this model.   Thus the method does apply with this data.
*		> Primary limit of 60 second lookback is determining how far to look back - how much data is too much 
*			> Code developed looks at different time periods for the lookback to determine which time period best predicts the high_price
*			> There is no real way to determine the right lookback window - it is a matter of testing
* If you stuck with the neural network, what did you change to make it better? Did you change the architecture, did you change the optimizer? The learning rate? The activation function(s)? Why was the model stuck at `0` with an incredibly high root mean squared error?
*		> I would not be stuck with a NN - I have developed NN's for most of my career and believe they are valuable estimators for most problems
*		> First for any NN the data must be normalized - even though values in this data are in a similar range > best pratice and standard practice is to always normalize the data
*		> Without normalized data the model will work sometimes, then other times it will result in a very high RMSE and 0's for the predicted values > this is what was found when running the model "as was"
*		> Model setup was not correct:
*			> Need to specify the activation function for each layer and no activation function on the output layer
*			> Orginal setup had no activation functions on the hidden layers, and an activation function on the output layer - this will not work for a regression model
*			> Best optimizer for a Regression NN is RMSprop - produced the best results
*			> Tested mulitple different learning rates, a smaller rate of 0.005 worked the best
*				> As you can see from the code, my method allows me to put in various different learning rates and run tests to determine the optimal learning rate
*				> There is no right or wrong, learning rate depends on the data and should be tested using different values for the same data
*		> Model was stuck or not producing results given the setup and non-normalized data
* If you used a different model, why'd you choose this model? What about it made it work for this problem? Is this model complex and if so, is the complexity necessary? Is it intuitive enough to explain it to a lay-person? What was your optimizing metric? What were the hyperparameters and why'd you choose them?
*		> I tested a very basic regesssion XGBoost and the results were the same as the best NN Model
*			> By fine tuning the XGBoost hyperparameters it is likely that a better model could be achieved
*		> Also, I only tested two time periods for the XGBoost model - might want to explore different sizes of time period slices
* Did you include any regularization strategies in your model? If so, why'd you choose the one you did?
* Did you include visualizations? (everyone loves a good graphic)
*		> Some visualizations were included to understand the data and the final prdicted values
* How do we know the model is good? How understandable are the diagnostics? How will we know how good the model is predicting in production?
*		> The primary issue with this model setup is that there was not a hold out validation dataset
*		> A hold out sample should have been created, model should have predicted the values, and developed metrics to evulate the model based on the validation dataset
*		> This is how you would know if the model is perform well, and if the model is working in production
*			> A model in production should be close to the metrics of the validation data - measured on a regular bases
*				> Once the metrics diverge - it is time to re-build the model
* If we see data for more than a single day's worth of prices, how do expect the model to perform? Will it generalize well to new data? Will retraining with this new data be an issue for this model?
* **What question would you ask of the data, or add to this analysis that I haven't thought of?**

*  > One element I would note about my model is that I included the attribute price_rollmean > this attribute is highly correlated to the time slices of high_price 
*	> It is most likely not neccessary to have this attribute in the model
*	> Version shouold be tested without this attribute

*	> Note for other attributes > in code comments, highly correlated attributes were removed
*	> Also, attributes volume_traded and trades_count had very low variance compared to the target and no correlation, they were removed from the model inputs


## Note on Dependencies

Feel free to install dependencies however you wish (docker, pyenv, conda, etc.). This model was trained in a conda virtual environment, and you can find all of the details in the `spec-file.txt` (the output of `conda list`) file or the explicit packages in `spec-file-explicit.txt` (the output of `conda list --explicit`). You may upgrade/downgrade dependencies as you see fit, as long as it doesn't effect the solution.
