> [!NOTE]
> Fredik Nguyen, Sayaka Yamashita, Catalina Regolf, Clementine Lefloch &  Miquel Gomez

### Predicting Genre From Data – A Movie Classification Project
# Abstract
The world of movies is extensive and over a century old. The variety of movies that have been made is overwhelmingly large: whether it’s drama, action or romance, horror or sci-fi, there’s a movie out there that caters to each individual’s particular taste. When searching for what to watch, one of the first things we look at is genre — what kind of movie are we in the mood to watch? As such, having a movie tagged with the right genres is crucial for it to find its right audience. This sparked our interest in genre classification: what makes a movie belong to a particular genre? This project will focus on genre and the characteristics of movies that set genres apart. By the end of this project, we want to see if it’s possible to automatically assign genres to movies using their data. 

# Research questions
Our biggest goal for this project is to make a movie classifier that will assign genres to movies. As such, our research questions revolve around the different characteristics that movies have and their relationships with genre:
- Taking a look at numbers we can ask: Are genre and runtime related? How about the movie budget or its box office revenue? We can ask the same for release dates, movie languages and countries, age ratings, etc.
- Then there’s the stories that movies tell and how they determine their genre. How can we use movie summaries to extract their genres? What about the characters that appear in movies? Is there a trend for the kind of characters that appear in movies for each genre?
- How can we use all this data to build an accurate classifier?

# Additional datasets
In order to find more useful features that will help classify movies, we would like to make use of an additional dataset.

The TMDb dataset is a movie dataset that contains information about a million movies. We want to use it to enrich our original dataset with additional information that could be useful for our project like the budget of movies, age rating (whether it’s an adult movie or not), votes, etc. We will merge the datasets so that we only use the movies of the original dataset.

# Methods
**Data wrangling**
We load our databases and merge them by movie title and year so that we keep the movies from the CMU database with the additional information of the TMDb. We also fill in some missing values of the original dataset (runtime, revenue) with values from TMDb. We deal with duplicate movies and clean the data. 

The original database made use of over 300 different genres, so we grouped them into 39 new genre categories that the model will use and predict from. In addition we get the hot-vector of the genre of each movie: 1 if the movie has the corresponding genre, 0 o.w.

**Data enrichment**
From the plot, and the characters we extract additional information that can, potentially, improve our model:
- From the plots we get the sentiment analysis i.e. how positive or negative a movie is. And also related theme words; we get the proportion of words related to “love”, “death” and other topics that can enhance the model predictions.
- From the characters we get how many Male and Female actors there are and their age (grouped into age ranges: 0-20, 20-30…).

**Data analysis**
We make bar plots, scatter plots and heatmaps to visualize the relationship between movie genres and different movie characteristics. 

**Genre Predictor Model**
After getting all the data and the new features, we can start creating the model to predict the movie genres.

First of all we define the model: a Neural Network with (for now, to experiment) 5 layers, each of its corresponding size for input and output. Also we define some optimizers and the training functions.

We feed the model with batches of data in each iteration and use the loss to backpropagate the feedback. After all the iterations we plot the metrics *(Precision, Recall and F-score).*

Finally, with the trained model, we can test how it does with the “testing set”. The output of which is going to be in a separate file where we can see the movie, its original genre, and the new ones. 

Right now, the performance of the model is not near to being “good”. But this is normal since we are just preparing the skeleton and the bases of it. In the following iteration of the model we will use techniques like cross validation or grid search to get the best hyperparameters.

# Proposed timeline and team organization
By the 15th of November we will have established our main goals and ideas, and we will have started working towards them. The next two weeks will be spent reflecting on our project and refining our ideas and methods (and doing Homework 2!).

By the 29th of November we will be ready to spend the following 3 weeks giving it our all to complete our project! We want to work by achieving the following internal milestones:
**1. Data cleaning and data analysis (December 6th)**
- We will have done some initial data analysis and data cleaning by the Milestone P2 due date, but we might have to develop them further as we carry on working on the project.
**2. Movie classifier (December 13th)**
- Finishing the classifier will be an important milestone of our project. We will have reached this milestone once we obtain a classifier that works to our satisfaction.
**3. Data story (December 16th)**
- This milestone will be completed once we obtain the graphics we want to use and write the story we want to tell. 
**4. Blog post / article (December 19th)**
- The final milestone will be putting everything together in the blog post and completing our final hand-in.
**5. Milestone P3! (December 20th)**
- Final check that everything is in order. Our project will be complete!

These are tentative dates that we will try to follow so that our team achieves a good working rhythm. We will divide our work so that different teammates can work on different milestones and contribute to the completion of the project.



### How to use the library
Tell us how the code is arranged, any explanations goes here.


## Project Structure

The directory structure of new project looks like this:

```
├── data                        <- Project data files
│
├── src                         <- Source code
│   ├── data                            <- Data directory
│   ├── models                          <- Model directory
│   ├── utils                           <- Utility directory
│   ├── scripts                         <- Shell scripts
│
├── tests                       <- Tests of any kind
│
├── results.ipynb               <- a well-structured notebook showing the results
│
├── .gitignore                  <- List of files ignored by git
├── pip_requirements.txt        <- File for installing python dependencies
└── README.md
```

