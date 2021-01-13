# Predicting world happiness


## Background
The World Happiness Report is a landmark survey of the state of global happiness. The first report was published in 2012, the second in 2013, the third in 2015, and the fourth in the 2016 Update. The World Happiness 2017, which ranks 155 countries by their happiness levels, was released at the United Nations at an event celebrating International Day of Happiness on March 20th. The report continues to gain global recognition as governments, organizations and civil society increasingly use happiness indicators to inform their policy-making decisions. 

## Tools Used
- Pandas, Json: Data Wrangling
- Matpotllib, Seaborn: Data Visualization
- Folium, geopandas, country converter : Geospatial Analysis
- Scikit learn: Data Modelling


## Data
The world happiness dataset can be found on kaggle [here](https://www.kaggle.com/unsdsn/world-happiness).

The shape file used to create the map can be downloaded from [here](https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/50m/cultural/ne_50m_admin_0_countries.zip)

## Data Analysis
The data file contains 5 datasets 2015, 2016, 2017, 2018 and 2019. I merged all the datasets together to create a bigger dataset, and I performed my analysis on this.

From the exploratory data analysis carried out on the data, happiness score experienced a significant drop in 2017 but has countinued to rise till date; although I think 2020 might tell a whole different story.

The extent to which features like GDP per Capital, Family etc contribute to the happiness score has also increased, this is probably due to better understanding of how to handle the data.

I also did a geospatial analysis and realised that most of the countries with the highest happiness scores are in Europe whereas African countries have the lowest happiness scores. I used a Choropleth map that shows the happiness scores of different countries.

## Modelling

The RandomForestRegressor model used had an MSE of 0.2 and R2 Score of 82.65%; on the test set. The MSE score shows how little the margin of error of predictions is which is an excellent sign that the predictions are reliable.

Go through the notebook to see more exciting stuff!👌
