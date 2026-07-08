{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Supervised Learning\n",
    "## Project: Finding Donors for *CharityML*"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "In this notebook, some template code has already been provided for you, and it will be your job to implement the additional functionality necessary to successfully complete this project. Sections that begin with **'Implementation'** in the header indicate that the following block of code will require additional functionality which you must provide. Instructions will be provided for each section and the specifics of the implementation are marked in the code block with a `'TODO'` statement. Please be sure to read the instructions carefully!\n",
    "\n",
    "In addition to implementing code, there will be questions that you must answer which relate to the project and your implementation. Each section where you will answer a question is preceded by a **'Question X'** header. Carefully read each question and provide thorough answers in the following text boxes that begin with **'Answer:'**. Your project submission will be evaluated based on your answers to each of the questions and the implementation you provide.  \n",
    "\n",
    ">**Note:** Please specify WHICH VERSION OF PYTHON you are using when submitting this notebook. Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Version of Python: Python 3"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Getting Started\n",
    "\n",
    "In this project, you will employ several supervised algorithms of your choice to accurately model individuals' income using data collected from the 1994 U.S. Census. You will then choose the best candidate algorithm from preliminary results and further optimize this algorithm to best model the data. Your goal with this implementation is to construct a model that accurately predicts whether an individual makes more than $50,000. This sort of task can arise in a non-profit setting, where organizations survive on donations.  Understanding an individual's income can help a non-profit better understand how large of a donation to request, or whether or not they should reach out to begin with.  While it can be difficult to determine an individual's general income bracket directly from public sources, we can (as we will see) infer this value from other publically available features. \n",
    "\n",
    "The dataset for this project originates from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Census+Income). The datset was donated by Ron Kohavi and Barry Becker, after being published in the article _\"Scaling Up the Accuracy of Naive-Bayes Classifiers: A Decision-Tree Hybrid\"_. You can find the article by Ron Kohavi [online](https://www.aaai.org/Papers/KDD/1996/KDD96-033.pdf). The data we investigate here consists of small changes to the original dataset, such as removing the `'fnlwgt'` feature and records with missing or ill-formatted entries."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "----\n",
    "## Exploring the Data\n",
    "Run the code cell below to load necessary Python libraries and load the census data. Note that the last column from this dataset, `'income'`, will be our target label (whether an individual makes more than, or at most, $50,000 annually). All other columns are features about each individual in the census database."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>age</th>\n",
       "      <th>workclass</th>\n",
       "      <th>education_level</th>\n",
       "      <th>education-num</th>\n",
       "      <th>marital-status</th>\n",
       "      <th>occupation</th>\n",
       "      <th>relationship</th>\n",
       "      <th>race</th>\n",
       "      <th>sex</th>\n",
       "      <th>capital-gain</th>\n",
       "      <th>capital-loss</th>\n",
       "      <th>hours-per-week</th>\n",
       "      <th>native-country</th>\n",
       "      <th>income</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>39</td>\n",
       "      <td>State-gov</td>\n",
       "      <td>Bachelors</td>\n",
       "      <td>13.0</td>\n",
       "      <td>Never-married</td>\n",
       "      <td>Adm-clerical</td>\n",
       "      <td>Not-in-family</td>\n",
       "      <td>White</td>\n",
       "      <td>Male</td>\n",
       "      <td>2174.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>40.0</td>\n",
       "      <td>United-States</td>\n",
       "      <td>&lt;=50K</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   age   workclass education_level  education-num  marital-status  \\\n",
       "0   39   State-gov       Bachelors           13.0   Never-married   \n",
       "\n",
       "      occupation    relationship    race    sex  capital-gain  capital-loss  \\\n",
       "0   Adm-clerical   Not-in-family   White   Male        2174.0           0.0   \n",
       "\n",
       "   hours-per-week  native-country income  \n",
       "0            40.0   United-States  <=50K  "
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Import libraries necessary for this project\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "from time import time\n",
    "from IPython.display import display # Allows the use of display() for DataFrames\n",
    "\n",
    "# Import supplementary visualization code visuals.py\n",
    "import visuals as vs\n",
    "\n",
    "# Pretty display for notebooks\n",
    "%matplotlib inline\n",
    "\n",
    "# Load the Census dataset\n",
    "data = pd.read_csv(\"census.csv\")\n",
    "\n",
    "# Success - Display the first record\n",
    "display(data.head(n=1))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "data['income'] = data['income'].str.strip()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Implementation: Data Exploration\n",
    "A cursory investigation of the dataset will determine how many individuals fit into either group, and will tell us about the percentage of these individuals making more than \\$50,000. In the code cell below, you will need to compute the following:\n",
    "- The total number of records, `'n_records'`\n",
    "- The number of individuals making more than \\$50,000 annually, `'n_greater_50k'`.\n",
    "- The number of individuals making at most \\$50,000 annually, `'n_at_most_50k'`.\n",
    "- The percentage of individuals making more than \\$50,000 annually, `'greater_percent'`.\n",
    "\n",
    "** HINT: ** You may need to look at the table above to understand how the `'income'` entries are formatted. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Total number of records: 45222\n",
      "Individuals making more than $50,000: 11208\n",
      "Individuals making at most $50,000: 34014\n",
      "Percentage of individuals making more than $50,000: 24.78%\n"
     ]
    }
   ],
   "source": [
    "# TODO: Total number of records\n",
    "n_records = len(data)\n",
    "\n",
    "# TODO: Number of records where individual's income is more than $50,000\n",
    "n_greater_50k=(data['income'] =='>50K').sum()\n",
    "\n",
    "# TODO: Number of records where individual's income is at most $50,000\n",
    "n_at_most_50k = (data['income'] =='<=50K').sum()\n",
    "\n",
    "# TODO: Percentage of individuals whose income is more than $50,000\n",
    "greater_percent = (n_greater_50k / n_records) * 100\n",
    "\n",
    "# Print the results\n",
    "print(\"Total number of records: {}\".format(n_records))\n",
    "print(\"Individuals making more than $50,000: {}\".format(n_greater_50k))\n",
    "print(\"Individuals making at most $50,000: {}\".format(n_at_most_50k))\n",
    "print(\"Percentage of individuals making more than $50,000: {:.2f}%\".format(greater_percent))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "** Featureset Exploration **\n",
    "\n",
    "* **age**: continuous. \n",
    "* **workclass**: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked. \n",
    "* **education**: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool. \n",
    "* **education-num**: continuous. \n",
    "* **marital-status**: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse. \n",
    "* **occupation**: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces. \n",
    "* **relationship**: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried. \n",
    "* **race**: Black, White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other. \n",
    "* **sex**: Female, Male. \n",
    "* **capital-gain**: continuous. \n",
    "* **capital-loss**: continuous. \n",
    "* **hours-per-week**: continuous. \n",
    "* **native-country**: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "----\n",
    "## Preparing the Data\n",
    "Before data can be used as input for machine learning algorithms, it often must be cleaned, formatted, and restructured — this is typically known as **preprocessing**. Fortunately, for this dataset, there are no invalid or missing entries we must deal with, however, there are some qualities about certain features that must be adjusted. This preprocessing can help tremendously with the outcome and predictive power of nearly all learning algorithms."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Transforming Skewed Continuous Features\n",
    "A dataset may sometimes contain at least one feature whose values tend to lie near a single number, but will also have a non-trivial number of vastly larger or smaller values than that single number.  Algorithms can be sensitive to such distributions of values and can underperform if the range is not properly normalized. With the census dataset two features fit this description: '`capital-gain'` and `'capital-loss'`. \n",
    "\n",
    "Run the code cell below to plot a histogram of these two features. Note the range of the values present and how they are distributed."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7d012e465e10>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Split the data into features and target label\n",
    "income_raw = data['income']\n",
    "features_raw = data.drop('income', axis = 1)\n",
    "\n",
    "# Visualize skewed continuous features of original data\n",
    "vs.distribution(data)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "For highly-skewed feature distributions such as `'capital-gain'` and `'capital-loss'`, it is common practice to apply a <a href=\"https://en.wikipedia.org/wiki/Data_transformation_(statistics)\">logarithmic transformation</a> on the data so that the very large and very small values do not negatively affect the performance of a learning algorithm. Using a logarithmic transformation significantly reduces the range of values caused by outliers. Care must be taken when applying this transformation however: The logarithm of `0` is undefined, so we must translate the values by a small amount above `0` to apply the the logarithm successfully.\n",
    "\n",
    "Run the code cell below to perform a transformation on the data and visualize the results. Again, note the range of values and how they are distributed. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7d012bf70dd8>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Log-transform the skewed features\n",
    "skewed = ['capital-gain', 'capital-loss']\n",
    "features_log_transformed = pd.DataFrame(data = features_raw)\n",
    "features_log_transformed[skewed] = features_raw[skewed].apply(lambda x: np.log(x + 1))\n",
    "\n",
    "# Visualize the new log distributions\n",
    "vs.distribution(features_log_transformed, transformed = True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Normalizing Numerical Features\n",
    "In addition to performing transformations on features that are highly skewed, it is often good practice to perform some type of scaling on numerical features. Applying a scaling to the data does not change the shape of each feature's distribution (such as `'capital-gain'` or `'capital-loss'` above); however, normalization ensures that each feature is treated equally when applying supervised learners. Note that once scaling is applied, observing the data in its raw form will no longer have the same original meaning, as exampled below.\n",
    "\n",
    "Run the code cell below to normalize each numerical feature. We will use [`sklearn.preprocessing.MinMaxScaler`](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html) for this."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>age</th>\n",
       "      <th>workclass</th>\n",
       "      <th>education_level</th>\n",
       "      <th>education-num</th>\n",
       "      <th>marital-status</th>\n",
       "      <th>occupation</th>\n",
       "      <th>relationship</th>\n",
       "      <th>race</th>\n",
       "      <th>sex</th>\n",
       "      <th>capital-gain</th>\n",
       "      <th>capital-loss</th>\n",
       "      <th>hours-per-week</th>\n",
       "      <th>native-country</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0.301370</td>\n",
       "      <td>State-gov</td>\n",
       "      <td>Bachelors</td>\n",
       "      <td>0.800000</td>\n",
       "      <td>Never-married</td>\n",
       "      <td>Adm-clerical</td>\n",
       "      <td>Not-in-family</td>\n",
       "      <td>White</td>\n",
       "      <td>Male</td>\n",
       "      <td>0.667492</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.397959</td>\n",
       "      <td>United-States</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0.452055</td>\n",
       "      <td>Self-emp-not-inc</td>\n",
       "      <td>Bachelors</td>\n",
       "      <td>0.800000</td>\n",
       "      <td>Married-civ-spouse</td>\n",
       "      <td>Exec-managerial</td>\n",
       "      <td>Husband</td>\n",
       "      <td>White</td>\n",
       "      <td>Male</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.122449</td>\n",
       "      <td>United-States</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0.287671</td>\n",
       "      <td>Private</td>\n",
       "      <td>HS-grad</td>\n",
       "      <td>0.533333</td>\n",
       "      <td>Divorced</td>\n",
       "      <td>Handlers-cleaners</td>\n",
       "      <td>Not-in-family</td>\n",
       "      <td>White</td>\n",
       "      <td>Male</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.397959</td>\n",
       "      <td>United-States</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0.493151</td>\n",
       "      <td>Private</td>\n",
       "      <td>11th</td>\n",
       "      <td>0.400000</td>\n",
       "      <td>Married-civ-spouse</td>\n",
       "      <td>Handlers-cleaners</td>\n",
       "      <td>Husband</td>\n",
       "      <td>Black</td>\n",
       "      <td>Male</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.397959</td>\n",
       "      <td>United-States</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0.150685</td>\n",
       "      <td>Private</td>\n",
       "      <td>Bachelors</td>\n",
       "      <td>0.800000</td>\n",
       "      <td>Married-civ-spouse</td>\n",
       "      <td>Prof-specialty</td>\n",
       "      <td>Wife</td>\n",
       "      <td>Black</td>\n",
       "      <td>Female</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.397959</td>\n",
       "      <td>Cuba</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "        age          workclass education_level  education-num  \\\n",
       "0  0.301370          State-gov       Bachelors       0.800000   \n",
       "1  0.452055   Self-emp-not-inc       Bachelors       0.800000   \n",
       "2  0.287671            Private         HS-grad       0.533333   \n",
       "3  0.493151            Private            11th       0.400000   \n",
       "4  0.150685            Private       Bachelors       0.800000   \n",
       "\n",
       "        marital-status          occupation    relationship    race      sex  \\\n",
       "0        Never-married        Adm-clerical   Not-in-family   White     Male   \n",
       "1   Married-civ-spouse     Exec-managerial         Husband   White     Male   \n",
       "2             Divorced   Handlers-cleaners   Not-in-family   White     Male   \n",
       "3   Married-civ-spouse   Handlers-cleaners         Husband   Black     Male   \n",
       "4   Married-civ-spouse      Prof-specialty            Wife   Black   Female   \n",
       "\n",
       "   capital-gain  capital-loss  hours-per-week  native-country  \n",
       "0      0.667492           0.0        0.397959   United-States  \n",
       "1      0.000000           0.0        0.122449   United-States  \n",
       "2      0.000000           0.0        0.397959   United-States  \n",
       "3      0.000000           0.0        0.397959   United-States  \n",
       "4      0.000000           0.0        0.397959            Cuba  "
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Import sklearn.preprocessing.StandardScaler\n",
    "from sklearn.preprocessing import MinMaxScaler\n",
    "\n",
    "# Initialize a scaler, then apply it to the features\n",
    "scaler = MinMaxScaler() # default=(0, 1)\n",
    "numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']\n",
    "\n",
    "features_log_minmax_transform = pd.DataFrame(data = features_log_transformed)\n",
    "features_log_minmax_transform[numerical] = scaler.fit_transform(features_log_transformed[numerical])\n",
    "\n",
    "# Show an example of a record with scaling applied\n",
    "display(features_log_minmax_transform.head(n = 5))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>age</th>\n",
       "      <th>workclass</th>\n",
       "      <th>education_level</th>\n",
       "      <th>education-num</th>\n",
       "      <th>marital-status</th>\n",
       "      <th>occupation</th>\n",
       "      <th>relationship</th>\n",
       "      <th>race</th>\n",
       "      <th>sex</th>\n",
       "      <th>capital-gain</th>\n",
       "      <th>capital-loss</th>\n",
       "      <th>hours-per-week</th>\n",
       "      <th>native-country</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0.301370</td>\n",
       "      <td>State-gov</td>\n",
       "      <td>Bachelors</td>\n",
       "      <td>0.800000</td>\n",
       "      <td>Never-married</td>\n",
       "      <td>Adm-clerical</td>\n",
       "      <td>Not-in-family</td>\n",
       "      <td>White</td>\n",
       "      <td>Male</td>\n",
       "      <td>0.667492</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.397959</td>\n",
       "      <td>United-States</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0.452055</td>\n",
       "      <td>Self-emp-not-inc</td>\n",
       "      <td>Bachelors</td>\n",
       "      <td>0.800000</td>\n",
       "      <td>Married-civ-spouse</td>\n",
       "      <td>Exec-managerial</td>\n",
       "      <td>Husband</td>\n",
       "      <td>White</td>\n",
       "      <td>Male</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.122449</td>\n",
       "      <td>United-States</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0.287671</td>\n",
       "      <td>Private</td>\n",
       "      <td>HS-grad</td>\n",
       "      <td>0.533333</td>\n",
       "      <td>Divorced</td>\n",
       "      <td>Handlers-cleaners</td>\n",
       "      <td>Not-in-family</td>\n",
       "      <td>White</td>\n",
       "      <td>Male</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.397959</td>\n",
       "      <td>United-States</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0.493151</td>\n",
       "      <td>Private</td>\n",
       "      <td>11th</td>\n",
       "      <td>0.400000</td>\n",
       "      <td>Married-civ-spouse</td>\n",
       "      <td>Handlers-cleaners</td>\n",
       "      <td>Husband</td>\n",
       "      <td>Black</td>\n",
       "      <td>Male</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.397959</td>\n",
       "      <td>United-States</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0.150685</td>\n",
       "      <td>Private</td>\n",
       "      <td>Bachelors</td>\n",
       "      <td>0.800000</td>\n",
       "      <td>Married-civ-spouse</td>\n",
       "      <td>Prof-specialty</td>\n",
       "      <td>Wife</td>\n",
       "      <td>Black</td>\n",
       "      <td>Female</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.397959</td>\n",
       "      <td>Cuba</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "        age          workclass education_level  education-num  \\\n",
       "0  0.301370          State-gov       Bachelors       0.800000   \n",
       "1  0.452055   Self-emp-not-inc       Bachelors       0.800000   \n",
       "2  0.287671            Private         HS-grad       0.533333   \n",
       "3  0.493151            Private            11th       0.400000   \n",
       "4  0.150685            Private       Bachelors       0.800000   \n",
       "\n",
       "        marital-status          occupation    relationship    race      sex  \\\n",
       "0        Never-married        Adm-clerical   Not-in-family   White     Male   \n",
       "1   Married-civ-spouse     Exec-managerial         Husband   White     Male   \n",
       "2             Divorced   Handlers-cleaners   Not-in-family   White     Male   \n",
       "3   Married-civ-spouse   Handlers-cleaners         Husband   Black     Male   \n",
       "4   Married-civ-spouse      Prof-specialty            Wife   Black   Female   \n",
       "\n",
       "   capital-gain  capital-loss  hours-per-week  native-country  \n",
       "0      0.667492           0.0        0.397959   United-States  \n",
       "1      0.000000           0.0        0.122449   United-States  \n",
       "2      0.000000           0.0        0.397959   United-States  \n",
       "3      0.000000           0.0        0.397959   United-States  \n",
       "4      0.000000           0.0        0.397959            Cuba  "
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Import sklearn.preprocessing.StandardScaler\n",
    "from sklearn.preprocessing import MinMaxScaler\n",
    "\n",
    "# Initialize a scaler, then apply it to the features\n",
    "scaler = MinMaxScaler() # default=(0, 1)\n",
    "numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']\n",
    "\n",
    "features_log_minmax_transform = pd.DataFrame(data = features_log_transformed)\n",
    "features_log_minmax_transform[numerical] = scaler.fit_transform(features_log_transformed[numerical])\n",
    "\n",
    "# Show an example of a record with scaling applied\n",
    "display(features_log_minmax_transform.head(n = 5))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Implementation: Data Preprocessing\n",
    "\n",
    "From the table in **Exploring the Data** above, we can see there are several features for each record that are non-numeric. Typically, learning algorithms expect input to be numeric, which requires that non-numeric features (called *categorical variables*) be converted. One popular way to convert categorical variables is by using the **one-hot encoding** scheme. One-hot encoding creates a _\"dummy\"_ variable for each possible category of each non-numeric feature. For example, assume `someFeature` has three possible entries: `A`, `B`, or `C`. We then encode this feature into `someFeature_A`, `someFeature_B` and `someFeature_C`.\n",
    "\n",
    "|   | someFeature |                    | someFeature_A | someFeature_B | someFeature_C |\n",
    "| :-: | :-: |                            | :-: | :-: | :-: |\n",
    "| 0 |  B  |  | 0 | 1 | 0 |\n",
    "| 1 |  C  | ----> one-hot encode ----> | 0 | 0 | 1 |\n",
    "| 2 |  A  |  | 1 | 0 | 0 |\n",
    "\n",
    "Additionally, as with the non-numeric features, we need to convert the non-numeric target label, `'income'` to numerical values for the learning algorithm to work. Since there are only two possible categories for this label (\"<=50K\" and \">50K\"), we can avoid using one-hot encoding and simply encode these two categories as `0` and `1`, respectively. In code cell below, you will need to implement the following:\n",
    " - Use [`pandas.get_dummies()`](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html?highlight=get_dummies#pandas.get_dummies) to perform one-hot encoding on the `'features_log_minmax_transform'` data.\n",
    " - Convert the target label `'income_raw'` to numerical entries.\n",
    "   - Set records with \"<=50K\" to `0` and records with \">50K\" to `1`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "103 total features after one-hot encoding.\n",
      "['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week', 'workclass_ Federal-gov', 'workclass_ Local-gov', 'workclass_ Private', 'workclass_ Self-emp-inc', 'workclass_ Self-emp-not-inc', 'workclass_ State-gov', 'workclass_ Without-pay', 'education_level_ 10th', 'education_level_ 11th', 'education_level_ 12th', 'education_level_ 1st-4th', 'education_level_ 5th-6th', 'education_level_ 7th-8th', 'education_level_ 9th', 'education_level_ Assoc-acdm', 'education_level_ Assoc-voc', 'education_level_ Bachelors', 'education_level_ Doctorate', 'education_level_ HS-grad', 'education_level_ Masters', 'education_level_ Preschool', 'education_level_ Prof-school', 'education_level_ Some-college', 'marital-status_ Divorced', 'marital-status_ Married-AF-spouse', 'marital-status_ Married-civ-spouse', 'marital-status_ Married-spouse-absent', 'marital-status_ Never-married', 'marital-status_ Separated', 'marital-status_ Widowed', 'occupation_ Adm-clerical', 'occupation_ Armed-Forces', 'occupation_ Craft-repair', 'occupation_ Exec-managerial', 'occupation_ Farming-fishing', 'occupation_ Handlers-cleaners', 'occupation_ Machine-op-inspct', 'occupation_ Other-service', 'occupation_ Priv-house-serv', 'occupation_ Prof-specialty', 'occupation_ Protective-serv', 'occupation_ Sales', 'occupation_ Tech-support', 'occupation_ Transport-moving', 'relationship_ Husband', 'relationship_ Not-in-family', 'relationship_ Other-relative', 'relationship_ Own-child', 'relationship_ Unmarried', 'relationship_ Wife', 'race_ Amer-Indian-Eskimo', 'race_ Asian-Pac-Islander', 'race_ Black', 'race_ Other', 'race_ White', 'sex_ Female', 'sex_ Male', 'native-country_ Cambodia', 'native-country_ Canada', 'native-country_ China', 'native-country_ Columbia', 'native-country_ Cuba', 'native-country_ Dominican-Republic', 'native-country_ Ecuador', 'native-country_ El-Salvador', 'native-country_ England', 'native-country_ France', 'native-country_ Germany', 'native-country_ Greece', 'native-country_ Guatemala', 'native-country_ Haiti', 'native-country_ Holand-Netherlands', 'native-country_ Honduras', 'native-country_ Hong', 'native-country_ Hungary', 'native-country_ India', 'native-country_ Iran', 'native-country_ Ireland', 'native-country_ Italy', 'native-country_ Jamaica', 'native-country_ Japan', 'native-country_ Laos', 'native-country_ Mexico', 'native-country_ Nicaragua', 'native-country_ Outlying-US(Guam-USVI-etc)', 'native-country_ Peru', 'native-country_ Philippines', 'native-country_ Poland', 'native-country_ Portugal', 'native-country_ Puerto-Rico', 'native-country_ Scotland', 'native-country_ South', 'native-country_ Taiwan', 'native-country_ Thailand', 'native-country_ Trinadad&Tobago', 'native-country_ United-States', 'native-country_ Vietnam', 'native-country_ Yugoslavia']\n"
     ]
    }
   ],
   "source": [
    "# TODO: One-hot encode the 'features_log_minmax_transform' data using pandas.get_dummies()\n",
    "features_final = pd.get_dummies(features_log_minmax_transform)\n",
    "\n",
    "# TODO: Encode the 'income_raw' data to numerical values\n",
    "income = income_raw.apply(lambda x: 1 if x == '>50K' else 0)\n",
    "\n",
    "# Print the number of features after one-hot encoding\n",
    "encoded = list(features_final.columns)\n",
    "print(\"{} total features after one-hot encoding.\".format(len(encoded)))\n",
    "\n",
    "# Uncomment the following line to see the encoded feature names\n",
    "print(encoded)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Shuffle and Split Data\n",
    "Now all _categorical variables_ have been converted into numerical features, and all numerical features have been normalized. As always, we will now split the data (both features and their labels) into training and test sets. 80% of the data will be used for training and 20% for testing.\n",
    "\n",
    "Run the code cell below to perform this split."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training set has 36177 samples.\n",
      "Testing set has 9045 samples.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/opt/conda/lib/python3.6/site-packages/sklearn/cross_validation.py:41: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. Also note that the interface of the new CV iterators are different from that of this module. This module will be removed in 0.20.\n",
      "  \"This module will be removed in 0.20.\", DeprecationWarning)\n"
     ]
    }
   ],
   "source": [
    "# Import train_test_split\n",
    "from sklearn.cross_validation import train_test_split\n",
    "\n",
    "# Split the 'features' and 'income' data into training and testing sets\n",
    "X_train, X_test, y_train, y_test = train_test_split(features_final, \n",
    "                                                    income, \n",
    "                                                    test_size = 0.2, \n",
    "                                                    random_state = 0)\n",
    "\n",
    "# Show the results of the split\n",
    "print(\"Training set has {} samples.\".format(X_train.shape[0]))\n",
    "print(\"Testing set has {} samples.\".format(X_test.shape[0]))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "*Note: this Workspace is running on `sklearn` v0.19. If you use the newer version (>=\"0.20\"), the `sklearn.cross_validation` has been replaced with `sklearn.model_selection`.*"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "----\n",
    "## Evaluating Model Performance\n",
    "In this section, we will investigate four different algorithms, and determine which is best at modeling the data. Three of these algorithms will be supervised learners of your choice, and the fourth algorithm is known as a *naive predictor*."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Metrics and the Naive Predictor\n",
    "*CharityML*, equipped with their research, knows individuals that make more than \\$50,000 are most likely to donate to their charity. Because of this, *CharityML* is particularly interested in predicting who makes more than \\$50,000 accurately. It would seem that using **accuracy** as a metric for evaluating a particular model's performace would be appropriate. Additionally, identifying someone that *does not* make more than \\$50,000 as someone who does would be detrimental to *CharityML*, since they are looking to find individuals willing to donate. Therefore, a model's ability to precisely predict those that make more than \\$50,000 is *more important* than the model's ability to **recall** those individuals. We can use **F-beta score** as a metric that considers both precision and recall:\n",
    "\n",
    "$$ F_{\\beta} = (1 + \\beta^2) \\cdot \\frac{precision \\cdot recall}{\\left( \\beta^2 \\cdot precision \\right) + recall} $$\n",
    "\n",
    "In particular, when $\\beta = 0.5$, more emphasis is placed on precision. This is called the **F$_{0.5}$ score** (or F-score for simplicity).\n",
    "\n",
    "Looking at the distribution of classes (those who make at most $50,000,  and those who make more), it is clear most individuals do not make more than $50,000. This can greatly affect **accuracy**, since we could simply say *\"this person does not make more than \\$50,000\"* and generally be right, without ever looking at the data! Making such a statement would be called **naive**, since we have not considered any information to substantiate the claim. It is always important to consider the *naive prediction* for your data, to help establish a benchmark for whether a model is performing well. That been said, using that prediction would be pointless: If we predicted all people made less than \\$50,000, *CharityML* would identify no one as donors. \n",
    "\n",
    "\n",
    "#### Note: Recap of accuracy, precision, recall\n",
    "\n",
    "** Accuracy ** measures how often the classifier makes the correct prediction. It’s the ratio of the number of correct predictions to the total number of predictions (the number of test data points).\n",
    "\n",
    "** Precision ** tells us what proportion of messages we classified as spam, actually were spam.\n",
    "It is a ratio of true positives(words classified as spam, and which are actually spam) to all positives(all words classified as spam, irrespective of whether that was the correct classificatio), in other words it is the ratio of\n",
    "\n",
    "`[True Positives/(True Positives + False Positives)]`\n",
    "\n",
    "** Recall(sensitivity)** tells us what proportion of messages that actually were spam were classified by us as spam.\n",
    "It is a ratio of true positives(words classified as spam, and which are actually spam) to all the words that were actually spam, in other words it is the ratio of\n",
    "\n",
    "`[True Positives/(True Positives + False Negatives)]`\n",
    "\n",
    "For classification problems that are skewed in their classification distributions like in our case, for example if we had a 100 text messages and only 2 were spam and the rest 98 weren't, accuracy by itself is not a very good metric. We could classify 90 messages as not spam(including the 2 that were spam but we classify them as not spam, hence they would be false negatives) and 10 as spam(all 10 false positives) and still get a reasonably good accuracy score. For such cases, precision and recall come in very handy. These two metrics can be combined to get the F1 score, which is weighted average(harmonic mean) of the precision and recall scores. This score can range from 0 to 1, with 1 being the best possible F1 score(we take the harmonic mean as we are dealing with ratios)."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Question 1 - Naive Predictor Performace\n",
    "* If we chose a model that always predicted an individual made more than $50,000, what would  that model's accuracy and F-score be on this dataset? You must use the code cell below and assign your results to `'accuracy'` and `'fscore'` to be used later.\n",
    "\n",
    "** Please note ** that the the purpose of generating a naive predictor is simply to show what a base model without any intelligence would look like. In the real world, ideally your base model would be either the results of a previous model or could be based on a research paper upon which you are looking to improve. When there is no benchmark model set, getting a result better than random choice is a place you could start from.\n",
    "\n",
    "** HINT: ** \n",
    "\n",
    "* When we have a model that always predicts '1' (i.e. the individual makes more than 50k) then our model will have no True Negatives(TN) or False Negatives(FN) as we are not making any negative('0' value) predictions. Therefore our Accuracy in this case becomes the same as our Precision(True Positives/(True Positives + False Positives)) as every prediction that we have made with value '1' that should have '0' becomes a False Positive; therefore our denominator in this case is the total number of records we have in total. \n",
    "* Our Recall score(True Positives/(True Positives + False Negatives)) in this setting becomes 1 as we have no False Negatives."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Naive Predictor: [Accuracy score: 0.2478, F-score: 0.2917]\n"
     ]
    }
   ],
   "source": [
    "\n",
    "TP = np.sum(income) # Counting the ones as this is the naive case. Note that 'income' is the 'income_raw' data \n",
    "#encoded to numerical values done in the data preprocessing step.\n",
    "FP = income.count() - TP # Specific to the naive case\n",
    "\n",
    "TN = 0 # No predicted negatives in the naive case\n",
    "FN = 0 # No predicted negatives in the naive case\n",
    "\n",
    "# TODO: Calculate accuracy, precision and recall\n",
    "accuracy = (TP + TN) / income.count()\n",
    "recall = TP / (TP + FN)\n",
    "precision = TP / (TP + FP)\n",
    "\n",
    "# TODO: Calculate F-score using the formula above for beta = 0.5 and correct values for precision and recall.\n",
    "beta = 0.5\n",
    "fscore = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)\n",
    "\n",
    "# Print the results \n",
    "print(\"Naive Predictor: [Accuracy score: {:.4f}, F-score: {:.4f}]\".format(accuracy, fscore))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "###  Supervised Learning Models\n",
    "**The following are some of the supervised learning models that are currently available in** [`scikit-learn`](http://scikit-learn.org/stable/supervised_learning.html) **that you may choose from:**\n",
    "- Gaussian Naive Bayes (GaussianNB)\n",
    "- Decision Trees\n",
    "- Ensemble Methods (Bagging, AdaBoost, Random Forest, Gradient Boosting)\n",
    "- K-Nearest Neighbors (KNeighbors)\n",
    "- Stochastic Gradient Descent Classifier (SGDC)\n",
    "- Support Vector Machines (SVM)\n",
    "- Logistic Regression"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Question 2 - Model Application\n",
    "List three of the supervised learning models above that are appropriate for this problem that you will test on the census data. For each model chosen\n",
    "\n",
    "- Describe one real-world application in industry where the model can be applied. \n",
    "- What are the strengths of the model; when does it perform well?\n",
    "- What are the weaknesses of the model; when does it perform poorly?\n",
    "- What makes this model a good candidate for the problem, given what you know about the data?\n",
    "\n",
    "** HINT: **\n",
    "\n",
    "Structure your answer in the same format as above^, with 4 parts for each of the three models you pick. Please include references with your answer."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Answer:**\n",
    "### 1.  Decision Trees\n",
    "\n",
    "  - **Real world application**: Used in credit approval, fraud detection, and decision_making systems.\n",
    "  \n",
    "  \n",
    "  - **Strengths**:Easy to interpret; captures non linear relationships, no need for feature scaling. \n",
    "  \n",
    "  \n",
    "  - **Weakness**:\n",
    "  Prone to overfitting; unstable (small data changes can change the tree a lot).\n",
    "  \n",
    "  \n",
    "  - **Why it fits this problem**: \n",
    "  The dataset has mixed feature types and complex patterns, which trees can naturally capture.\n",
    "  \n",
    "###  2. Ensemble Methods\n",
    " \n",
    " - **Real-world application**:\n",
    "Widely used in finance, healthcare, and recommendation systems.\n",
    "\n",
    " \n",
    " - **Strengths**:\n",
    "High accuracy; reduces overfitting (especially Random Forest); captures complex patterns.\n",
    "\n",
    "\n",
    " - **Weaknesses**:\n",
    "More computationally expensive; harder to interpret.\n",
    "\n",
    "\n",
    " - **Why it fits this problem**:\n",
    "Income prediction likely depends on complex interactions between features—ensembles handle this very well and usually perform best.\n",
    "\n",
    "###  3. Support Vector Machine\n",
    "\n",
    " - **Real-world application**:\n",
    "Used in image classification, bioinformatics, and text classification.\n",
    "\n",
    "\n",
    " - **Strengths**:\n",
    "Effective in high-dimensional spaces; can model complex boundaries with kernels.\n",
    "\n",
    "\n",
    " - **Weaknesses**:\n",
    "Slow on large datasets; sensitive to parameter tuning (C, gamma).\n",
    "\n",
    "\n",
    " - **Why it fits this problem**: \n",
    "Works well with high-dimensional data (after one-hot encoding), but may be computationally expensive here."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Implementation - Creating a Training and Predicting Pipeline\n",
    "To properly evaluate the performance of each model you've chosen, it's important that you create a training and predicting pipeline that allows you to quickly and effectively train models using various sizes of training data and perform predictions on the testing data. Your implementation here will be used in the following section.\n",
    "In the code block below, you will need to implement the following:\n",
    " - Import `fbeta_score` and `accuracy_score` from [`sklearn.metrics`](http://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics).\n",
    " - Fit the learner to the sampled training data and record the training time.\n",
    " - Perform predictions on the test data `X_test`, and also on the first 300 training points `X_train[:300]`.\n",
    "   - Record the total prediction time.\n",
    " - Calculate the accuracy score for both the training subset and testing set.\n",
    " - Calculate the F-score for both the training subset and testing set.\n",
    "   - Make sure that you set the `beta` parameter!"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "# TODO: Import two metrics from sklearn - fbeta_score and accuracy_score\n",
    "from sklearn.metrics import fbeta_score, accuracy_score \n",
    "def train_predict(learner, sample_size, X_train, y_train, X_test, y_test): \n",
    "    '''\n",
    "    inputs:\n",
    "       - learner: the learning algorithm to be trained and predicted on\n",
    "       - sample_size: the size of samples (number) to be drawn from training set\n",
    "       - X_train: features training set\n",
    "       - y_train: income training set\n",
    "       - X_test: features testing set\n",
    "       - y_test: income testing set\n",
    "    '''\n",
    "    \n",
    "    results = {}\n",
    "    \n",
    "    # TODO: Fit the learner to the training data using slicing with 'sample_size' using .fit(training_features[:], training_labels[:])\n",
    "    start = time() # Get start time\n",
    "    learner = learner.fit(X_train[:sample_size],y_train[:sample_size])\n",
    "    end = time() # Get end time\n",
    "    \n",
    "    # TODO: Calculate the training time\n",
    "    results['train_time'] = end - start\n",
    "        \n",
    "    # TODO: Get the predictions on the test set(X_test),\n",
    "    #       then get predictions on the first 300 training samples(X_train) using .predict()\n",
    "    start = time() # Get start time\n",
    "    predictions_test = learner.predict(X_test)\n",
    "    predictions_train = learner.predict(X_train[:300])\n",
    "    end = time() # Get end time\n",
    "    \n",
    "    # TODO: Calculate the total prediction time\n",
    "    results['pred_time'] = end-start\n",
    "            \n",
    "    # TODO: Compute accuracy on the first 300 training samples which is y_train[:300]\n",
    "    results['acc_train'] = accuracy_score(y_train[:300], predictions_train)\n",
    "        \n",
    "    # TODO: Compute accuracy on test set using accuracy_score()\n",
    "    results['acc_test'] = accuracy_score(y_test, predictions_test)\n",
    "    \n",
    "    # TODO: Compute F-score on the the first 300 training samples using fbeta_score()\n",
    "    results['f_train'] = fbeta_score(y_train[:300], predictions_train, beta=0.5)\n",
    "        \n",
    "    # TODO: Compute F-score on the test set which is y_test\n",
    "    results['f_test'] = fbeta_score(y_test, predictions_test, beta=0.5)\n",
    "       \n",
    "    # Success\n",
    "    print(\"{} trained on {} samples.\".format(learner.__class__.__name__, sample_size))\n",
    "        \n",
    "    # Return the results\n",
    "    return results"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Implementation: Initial Model Evaluation\n",
    "In the code cell, you will need to implement the following:\n",
    "- Import the three supervised learning models you've discussed in the previous section.\n",
    "- Initialize the three models and store them in `'clf_A'`, `'clf_B'`, and `'clf_C'`.\n",
    "  - Use a `'random_state'` for each model you use, if provided.\n",
    "  - **Note:** Use the default settings for each model — you will tune one specific model in a later section.\n",
    "- Calculate the number of records equal to 1%, 10%, and 100% of the training data.\n",
    "  - Store those values in `'samples_1'`, `'samples_10'`, and `'samples_100'` respectively.\n",
    "\n",
    "**Note:** Depending on which algorithms you chose, the following implementation may take some time to run!"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "DecisionTreeClassifier trained on 361 samples.\n",
      "DecisionTreeClassifier trained on 3617 samples.\n",
      "DecisionTreeClassifier trained on 36177 samples.\n",
      "SVC trained on 361 samples.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/opt/conda/lib/python3.6/site-packages/sklearn/metrics/classification.py:1135: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 due to no predicted samples.\n",
      "  'precision', 'predicted', average, warn_for)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "SVC trained on 3617 samples.\n",
      "SVC trained on 36177 samples.\n",
      "RandomForestClassifier trained on 361 samples.\n",
      "RandomForestClassifier trained on 3617 samples.\n",
      "RandomForestClassifier trained on 36177 samples.\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7d0156e74780>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# TODO: Import the three supervised learning models from sklearn\n",
    "from sklearn.tree import DecisionTreeClassifier\n",
    "from sklearn.svm import SVC \n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "# TODO: Initialize the three models\n",
    "clf_A = DecisionTreeClassifier(random_state = 42)\n",
    "clf_B = SVC(random_state= 42)\n",
    "clf_C = RandomForestClassifier(random_state = 42)\n",
    "\n",
    "# TODO: Calculate the number of samples for 1%, 10%, and 100% of the training data\n",
    "# HINT: samples_100 is the entire training set i.e. len(y_train)\n",
    "# HINT: samples_10 is 10% of samples_100 (ensure to set the count of the values to be `int` and not `float`)\n",
    "# HINT: samples_1 is 1% of samples_100 (ensure to set the count of the values to be `int` and not `float`)\n",
    "samples_100 = len(y_train)\n",
    "samples_10 = int(samples_100 * 0.1)\n",
    "samples_1 = int(samples_100 * 0.01)\n",
    "\n",
    "# Collect results on the learners\n",
    "results = {}\n",
    "for clf in [clf_A, clf_B, clf_C]:\n",
    "    clf_name = clf.__class__.__name__\n",
    "    results[clf_name] = {}\n",
    "    for i, samples in enumerate([samples_1, samples_10, samples_100]):\n",
    "        results[clf_name][i] = \\\n",
    "        train_predict(clf, samples, X_train, y_train, X_test, y_test)\n",
    "\n",
    "# Run metrics visualization for the three supervised learning models chosen\n",
    "vs.evaluate(results, accuracy, fscore)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "----\n",
    "## Improving Results\n",
    "In this final section, you will choose from the three supervised learning models the *best* model to use on the student data. You will then perform a grid search optimization for the model over the entire training set (`X_train` and `y_train`) by tuning at least one parameter to improve upon the untuned model's F-score. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Question 3 - Choosing the Best Model\n",
    "\n",
    "* Based on the evaluation you performed earlier, in one to two paragraphs, explain to *CharityML* which of the three models you believe to be most appropriate for the task of identifying individuals that make more than \\$50,000. \n",
    "\n",
    "** HINT: ** \n",
    "Look at the graph at the bottom left from the cell above(the visualization created by `vs.evaluate(results, accuracy, fscore)`) and check the F score for the testing set when 100% of the training set is used. Which model has the highest score? Your answer should include discussion of the:\n",
    "* metrics - F score on the testing when 100% of the training data is used, \n",
    "* prediction/training time\n",
    "* the algorithm's suitability for the data."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Answer: **\n",
    "\n",
    "Based on the evaluation results, the Random Forest Classifier is the most appropriate model for this task. When using 100% of the training data, Random Forest achieves the highest F-score on the testing set compared to Decision Tree and SVC, showing a better balance between precision and recall. This is especially important for this problem because correctly identifying individuals who make more than $50,000 is more critical than simply maximizing overall accuracy.\n",
    "\n",
    "In addition to its strong performance, Random Forest also demonstrates good generalization, as its training and testing scores are both high and relatively close, unlike the Decision Tree which shows signs of overfitting. While SVC provides reasonable performance, it is significantly slower in both training and prediction, making it less practical for larger datasets. Overall, Random Forest offers the best combination of performance, stability, and efficiency, making it the most suitable choice for this classification task."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Question 4 - Describing the Model in Layman's Terms\n",
    "\n",
    "* In one to two paragraphs, explain to *CharityML*, in layman's terms, how the final model chosen is supposed to work. Be sure that you are describing the major qualities of the model, such as how the model is trained and how the model makes a prediction. Avoid using advanced mathematical jargon, such as describing equations.\n",
    "\n",
    "** HINT: **\n",
    "\n",
    "When explaining your model, if using external resources please include all citations."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Answer: ** \n",
    "\n",
    "The final model chosen, Random Forest, works by combining many simple decision-making processes to make a stronger and more reliable prediction. Instead of relying on a single decision tree (which might make mistakes or overfit the data), Random Forest builds many different trees using slightly different subsets of the data. Each tree looks at features such as age, education, work type, and hours worked, and makes its own prediction about whether a person earns more than $50,000. The model then takes a “vote” across all the trees, and the majority decision becomes the final prediction.\n",
    "\n",
    "During training, the model learns patterns from past data by creating these multiple trees and identifying which combinations of features tend to be associated with higher income. When making a prediction on new data, it passes the information through all the trees and combines their outputs, which helps reduce errors and improves stability. This approach makes Random Forest both accurate and reliable, as it balances different perspectives instead of depending on a single rule, leading to better performance on unseen data."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Implementation: Model Tuning\n",
    "Fine tune the chosen model. Use grid search (`GridSearchCV`) with at least one important parameter tuned with at least 3 different values. You will need to use the entire training set for this. In the code cell below, you will need to implement the following:\n",
    "- Import [`sklearn.grid_search.GridSearchCV`](http://scikit-learn.org/0.17/modules/generated/sklearn.grid_search.GridSearchCV.html) and [`sklearn.metrics.make_scorer`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html).\n",
    "- Initialize the classifier you've chosen and store it in `clf`.\n",
    " - Set a `random_state` if one is available to the same state you set before.\n",
    "- Create a dictionary of parameters you wish to tune for the chosen model.\n",
    " - Example: `parameters = {'parameter' : [list of values]}`.\n",
    " - **Note:** Avoid tuning the `max_features` parameter of your learner if that parameter is available!\n",
    "- Use `make_scorer` to create an `fbeta_score` scoring object (with $\\beta = 0.5$).\n",
    "- Perform grid search on the classifier `clf` using the `'scorer'`, and store it in `grid_obj`.\n",
    "- Fit the grid search object to the training data (`X_train`, `y_train`), and store it in `grid_fit`.\n",
    "\n",
    "**Note:** Depending on the algorithm chosen and the parameter list, the following implementation may take some time to run!"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Unoptimized model\n",
      "------\n",
      "Accuracy score on testing data: 0.8431\n",
      "F-score on testing data: 0.6842\n",
      "\n",
      "Optimized Model\n",
      "------\n",
      "Final accuracy score on the testing data: 0.8619\n",
      "Final F-score on the testing data: 0.7334\n"
     ]
    }
   ],
   "source": [
    "# TODO: Import 'GridSearchCV', 'make_scorer', and any other necessary libraries\n",
    "from sklearn.model_selection import GridSearchCV\n",
    "from sklearn.metrics import make_scorer, fbeta_score, accuracy_score\n",
    "# TODO: Initialize the classifier\n",
    "clf = RandomForestClassifier(random_state=42)\n",
    "\n",
    "# TODO: Create the parameters list you wish to tune, using a dictionary if needed.\n",
    "# HINT: parameters = {'parameter_1': [value1, value2], 'parameter_2': [value1, value2]}\n",
    "parameters = {\n",
    "    'n_estimators': [50, 100, 200],\n",
    "    'min_samples_split': [2, 5, 10],\n",
    "    'min_samples_leaf': [1, 2, 4]\n",
    "}\n",
    "\n",
    "# TODO: Make an fbeta_score scoring object using make_scorer()\n",
    "scorer = make_scorer(fbeta_score, beta=0.5)\n",
    "\n",
    "# TODO: Perform grid search on the classifier using 'scorer' as the scoring method using GridSearchCV()\n",
    "grid_obj = GridSearchCV(clf, parameters, scoring=scorer)\n",
    "\n",
    "# TODO: Fit the grid search object to the training data and find the optimal parameters using fit()\n",
    "grid_fit = grid_obj.fit(X_train, y_train)\n",
    "\n",
    "# Get the estimator\n",
    "best_clf = grid_fit.best_estimator_\n",
    "\n",
    "# Make predictions using the unoptimized and model\n",
    "predictions = (clf.fit(X_train, y_train)).predict(X_test)\n",
    "best_predictions = best_clf.predict(X_test)\n",
    "\n",
    "# Report the before-and-afterscores\n",
    "print(\"Unoptimized model\\n------\")\n",
    "print(\"Accuracy score on testing data: {:.4f}\".format(accuracy_score(y_test, predictions)))\n",
    "print(\"F-score on testing data: {:.4f}\".format(fbeta_score(y_test, predictions, beta = 0.5)))\n",
    "print(\"\\nOptimized Model\\n------\")\n",
    "print(\"Final accuracy score on the testing data: {:.4f}\".format(accuracy_score(y_test, best_predictions)))\n",
    "print(\"Final F-score on the testing data: {:.4f}\".format(fbeta_score(y_test, best_predictions, beta = 0.5)))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Question 5 - Final Model Evaluation\n",
    "\n",
    "* What is your optimized model's accuracy and F-score on the testing data? \n",
    "* Are these scores better or worse than the unoptimized model? \n",
    "* How do the results from your optimized model compare to the naive predictor benchmarks you found earlier in **Question 1**?_  \n",
    "\n",
    "**Note:** Fill in the table below with your results, and then provide discussion in the **Answer** box."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Results:\n",
    "\n",
    "|     Metric     | Unoptimized Model | Optimized Model |\n",
    "| :------------: | :---------------: | :-------------: | \n",
    "| Accuracy Score | 0.8431                  | 0.8619                |\n",
    "| F-score  |0.6842                     | 0.7334               |   EXAMPLE       |\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Answer:**\n",
    "The optimized model performs better than the unoptimized model. The optimized model achieved an accuracy of 0.8619 and an F-score of 0.7334 on the testing data. In comparison, the unoptimized model had an accuracy of 0.8431 and an F-score of 0.6842.\n",
    "\n",
    "Both accuracy and F-score improved after tuning, with a noticeable increase in F-score, which is especially important in this problem because it balances precision and recall. This indicates that the optimized model is better at correctly identifying individuals who earn more than $50K while reducing false positive predictions.\n",
    "\n",
    "Compared to the naive predictor from Question 1, the optimized model performs significantly better. The naive predictor tends to have poor precision and an unbalanced performance, while the optimized model provides a much more reliable and meaningful prediction. Therefor,  the optimized model is more suitable for this classification task."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "----\n",
    "## Feature Importance\n",
    "\n",
    "An important task when performing supervised learning on a dataset like the census data we study here is determining which features provide the most predictive power. By focusing on the relationship between only a few crucial features and the target label we simplify our understanding of the phenomenon, which is most always a useful thing to do. In the case of this project, that means we wish to identify a small number of features that most strongly predict whether an individual makes at most or more than \\$50,000.\n",
    "\n",
    "Choose a scikit-learn classifier (e.g., adaboost, random forests) that has a `feature_importance_` attribute, which is a function that ranks the importance of features according to the chosen classifier.  In the next python cell fit this classifier to training set and use this attribute to determine the top 5 most important features for the census dataset."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Question 6 - Feature Relevance Observation\n",
    "When **Exploring the Data**, it was shown there are thirteen available features for each individual on record in the census data. Of these thirteen records, which five features do you believe to be most important for prediction, and in what order would you rank them and why?"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Answer:**\n",
    "\n",
    "The five important  features for  predicting whether an individual earns more than $50k are likely:\n",
    "\n",
    "1. **capital gain:** this featurehas a  very strong impact because individualswith high gains are much more likely to have higher income.\n",
    "\n",
    "\n",
    "2. **capital loss:** Although less frequent, capital loss still provides useful financial signals and helps differentiate income levels. \n",
    "\n",
    "\n",
    "3. **education num:** Higher education levels are strongly correlated with higher-paying jobs, making this an important predictor.\n",
    "\n",
    "\n",
    "4. **hours per week:**People who work more hours tend to earn more, so this feature contributes significantly to the prediction.\n",
    "\n",
    "\n",
    "5. **age**: Age reflects experience and career progression, which are important factors in determining income."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Implementation - Extracting Feature Importance\n",
    "Choose a `scikit-learn` supervised learning algorithm that has a `feature_importance_` attribute availble for it. This attribute is a function that ranks the importance of each feature when making predictions based on the chosen algorithm.\n",
    "\n",
    "In the code cell below, you will need to implement the following:\n",
    " - Import a supervised learning model from sklearn if it is different from the three used earlier.\n",
    " - Train the supervised model on the entire training set.\n",
    " - Extract the feature importances using `'.feature_importances_'`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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AFBEREQkZTbpPoAiUjErVqxgzpw5LFu2rLqLIlKmOnXq0KxZM2rVqlXdRRHZ7CgAFEmROXPmUL9+fYqKijCz6i6OSF7uzoIFC5gzZw7Nmzev7uKIbHbUBCySIsuWLaNRo0YK/mSjZ2Y0atRItdUiVUQBoEjKKPiTTYWOVZGqowBQREREJGXUB1Akxezhyq1h8XPL/m3xGjVqsP/++5cMP/3001T0F32+++47HnvsMS6++OKKFrFM7k6TJk2YMWMG2267LV988QU77bQTr732GgcffDAATZo04f3336dRo0Y58xg5ciTTpk3j2muvzbucMWPGcPvtt/PMM8+sM23AgAH06dOHevXqVc5KiYhkUQ2giBRU3bp1mTp1asnf+vyc43fffce9995b4flWrVpVZhoz48ADD2T8+PEAjBs3jrZt2zJu3DgAPvjgAxo3bpw3+APo0aNHqcFfWQYMGMDSpUvXe34RkbIoABSRardq1Squvvpq2rdvT+vWrbn//vsBWLx4MUcccQTt2rVj//3355///CcA1157LbNmzaJNmzZcffXVjBkzhuOPP74kv759+zJ48GAg/GRk//79Ofjgg3nqqaeYNWsW3bt354ADDuCQQw7h/fffX6c8nTt3Lgn4xo0bx69//eu1AsKDDjoIgPnz53PKKafQvn172rdvz9ixYwEYPHgwffv2BWDWrFl07NiR9u3b069fP7beeuuS5SxevJiePXuy9957c9ZZZ+HuDBw4kLlz53LYYYdx2GGHVeZmFhEpoSZgESmoH374gTZt2gDQvHlzRowYwV//+lcaNGjApEmTWL58OZ07d+boo49ml112YcSIEWyzzTZ8/fXXdOzYkR49enDrrbfy7rvvMnXqVCA0p5amTp06vP766wAcccQR3HfffbRo0YI33niDiy++mJdffnmt9AcddBD9+/cHYOLEifz+979nwIABQAgAO3fuDMBll13GFVdcwcEHH8ynn35Kt27dmD59+lp5XXbZZVx22WWcccYZ3HfffWtNe+utt3jvvffYaaed6Ny5M2PHjuXSSy/lz3/+M6NHj6Zx48brsYVFRMqmAFBECirTBJz0wgsv8M477zBs2DAAFi5cyIwZM2jWrBnXX389r776KltssQWff/45X331VYWXefrppwOhxm3cuHGceuqpJdOWL1++TvoOHTrw1ltvsWTJElasWMHWW2/N7rvvzsyZMxk3bhxXXnklAC+++CLTpk0rme/7779n0aJFa+U1fvx4nn76aQDOPPNMrrrqqrWW06xZMwDatGnD7NmzS/oZimyM7OGHqyxvP/fcKstb1qUAUESqnbvzl7/8hW7duq01fvDgwcyfP58pU6ZQq1YtioqKcr4XrmbNmqxevbpkODvNVlttBcDq1atp2LDhOgFotnr16rHnnnsyaNAg2rVrB0DHjh0ZNWoU8+bNo2XLliX5jR8/nrp161Z8pYHatWuXfK5RowYrV65cr3xERCpKfQBFpNp169aN//u//2PFihUAfPjhhyxZsoSFCxey/fbbU6tWLUaPHs0nn3wCQP369deqadttt92YNm0ay5cvZ+HChbz00ks5l7PNNtvQvHlznnrqKSAEnm+//XbOtJ07d2bAgAF06tQJgE6dOnHXXXfRsWPHkvfTHX300dx9990l8+QKLDt27Mjw4cMBGDp0aLm2R/b6iYhUNtUAiqRYeV7bUgjnn38+s2fPpl27diWvYXn66ac566yzOOGEEyguLqZNmzbsvffeADRq1IjOnTuz3377ccwxx3Dbbbdx2mmn0bp1a1q0aEHbtm3zLmvIkCFcdNFF3HTTTaxYsYJevXrxk5/8ZJ10nTt35q677ioJANu1a8ecOXM4//zzS9IMHDiQSy65hNatW7Ny5Uq6dOmyTj+/AQMGcPbZZ3PHHXdw3HHH0aBBgzK3R58+fTjmmGNo2rQpo0ePLtc2FBGpCHPfOL4A1kdxcbFPnjy5uoshssmYPn06++yzT3UXI1WWLl1K3bp1MTOGDh3K448/XvI0s5RNx+zGRX0AN35mNsXdi8tKpxpAEZEqNGXKFPr27Yu707BhQwYNGlTdRRIRUQAoIlKVDjnkkLz9DEVEqoseAhERERFJGQWAIiIiIimjAFBEREQkZRQAioiIiKSMHgIRSbHKfqVDeV7j8OWXX3L55ZczadIkateuTVFREQMGDGCvvfaq1LIkde3aldtvv53i4vxvRhgwYAB9+vShXr16ABx77LE89thjNGzYcIOWXVRURP369alRowYA9957LwcddFCF87nlllu4/vrrN6gs+bRt25a//e1vtGnThpUrV9KgQQPuv/9+zj77bAAOOOAAHnzwwZJfRck2efJkHnnkEQYOHJh3GbNnz+b444/n3XffXWfa4MGDOfroo9lpp50qZ4VEpEyqARSRgnF3Tj75ZLp27cqsWbOYNm0at9xyy3r9vm9lGzBgAEuXLi0ZHjVq1AYHfxmjR49m6tSpTJ06db2CPwgBYEWV96flDjroIMaNGwfA22+/TcuWLUuGlyxZwkcffZTzZdkZxcXFpQZ/ZRk8eDBz585d7/lFpOIUAIpIwYwePZpatWpx4YUXloxr06YNhxxyCGPGjOH4448vGd+3b18GDx4MhFq066+/nk6dOlFcXMybb75Jt27d2GOPPUp+eaO0+ZMuuugiiouLadWqFTfccAMQftFj7ty5HHbYYRx22GEly/z666+55ppruPfee0vmv/HGG7njjjsAuO2222jfvj2tW7cuyau88s170kknccABB9CqVSseeOABAK699lp++OEH2rRpw1lnncXs2bPZb7/9Sua5/fbbufHGG4FQ23n99ddz6KGHctdddzF//nxOOeUU2rdvT/v27Rk7duw6ZencuXNJwDdu3DguvPDCkp+1mzhxIu3ataNGjRosWbKEX/ziF7Rv3562bduWvNA6ue3nz5/PUUcdRbt27fjlL3/Jbrvtxtdffw3AqlWruOCCC2jVqhVHH300P/zwA8OGDWPy5MmcddZZtGnThh9++KFC21FE1o8CQBEpmHfffZcDDjhgvebdZZddGD9+PIcccgi9e/dm2LBhTJgwgX79+lUon5tvvpnJkyfzzjvv8Morr/DOO+9w6aWXstNOOzF69Oh1fnqtV69ePPHEEyXDTz75JKeeeiovvPACM2bMYOLEiUydOpUpU6bw6quv5lzmYYcdRps2bTjwwAMBSp130KBBTJkyhcmTJzNw4EAWLFjArbfeSt26dZk6dSpDhgwpcx2/++47XnnlFa688kouu+wyrrjiCiZNmsTw4cPX+im7jGQN4Lhx4+jSpQu1a9dm0aJFjBs3js6dO5dsu8MPP5xJkyYxevRorr76apYsWbJWXr///e85/PDDefPNNzn55JP59NNPS6bNmDGDSy65hPfee4+GDRsyfPhwevbsSXFxMUOGDGHq1KnUrVu3zPUTkQ1XsD6AZtYduAuoATzk7rfmSHMacCPgwNvufmahyiciG7cePXoAsP/++7N48WLq169P/fr1qVOnDt99912583nyySd54IEHWLlyJV988QXTpk2jdevWedO3bduWefPmMXfuXObPn8+2227LrrvuysCBA3nhhRdKfnd48eLFzJgxgy5duqyTx+jRo2ncuHHJ8AsvvJB33oEDBzJixAgAPvvsM2bMmEGjRo3KvX4Ap59+esnnF198kWnTppUMf//99yxatIj69euXjCsqKuLHH3/kyy+/5P3336dly5a0b9+eN954g3HjxvGrX/2qpNwjR47k9ttvB2DZsmVrBXgAr7/+ekn5u3fvzrbbblsyrXnz5rRp0wYI/Qpnz55dofUSkcpTkADQzGoA9wBHAXOASWY20t2nJdK0AK4DOrv7t2a2fSHKJiKF06pVK4YNG5ZzWs0/1hRtAAAgAElEQVSaNVm9enXJ8LJly9aaXrt2bQC22GKLks+Z4ZUrV5Y5P8DHH3/M7bffzqRJk9h2223p3bt3znTZevbsybBhw/jyyy/p1asXEPozXnfddfzyl78sc/5s+eYdM2YML774IuPHj6devXp07do1Z/nKWtetttqq5PPq1asZP358mTVrnTp1YtiwYTRt2hQzo2PHjowdO5aJEyfSsWPHknIPHz6cli1brjVvsg9nab8vn9xvNWrUUHOvSDUqVBNwB2Cmu3/k7j8CQ4ETs9JcANzj7t8CuPu8ApVNRArk8MMPZ/ny5Tz44IMl4yZNmsQrr7zCbrvtxrRp01i+fDkLFy7kpZdeqlDe5Zn/+++/Z6uttqJBgwZ89dVX/Pvf/y6ZVr9+fRYtWpQz7169ejF06FCGDRtGz549AejWrRuDBg1i8eLFAHz++efMm1e+y1a+eRcuXMi2225LvXr1eP/995kwYULJPLVq1WLFihUA7LDDDsybN48FCxawfPlynnnmmbzLOvroo7n77rtLhjN9+7J17tyZO++8k06dOgEhIHzkkUfYcccdSx6G6datG3/5y19Kgry33nprnXwOPvhgnnzySSDUGH777bdlbo/Str2IVI1CNQHvDHyWGJ4DHJiVZi8AMxtLaCa+0d2fy87IzPoAfQB23XXXKimsSFqU57UtlcnMGDFiBJdffjm33norderUKXkNzC677MJpp51G69atadGiRUnzaHmVZ/6f/OQntG3bllatWrH77ruX9G0D6NOnD8cccwxNmzZdpx9gq1atWLRoETvvvDNNmzYFQmA1ffr0koBp66235tFHH2X77ctuvMg3b/fu3bnvvvto3bo1LVu2LKl5y5SvdevW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jAJAERERkZRRACgiIiKSMgoARURERFJGAaCIiIhIyigAFBEREUkZBYAiIiIiKaMAUERERCRlFACKiIiIpIwCQBEREZGUUQAoIiIikjIKAEVERERSRgGgiIiISMooABQRERFJGQWAIiIiIimjAFBEREQkZRQAioiIiKRMzeougMjGyh626i5CQfi5Xt1FEBGRAitYDaCZdTezD8xsppldm2N6FzN708xWmlnPQpVLREREJG0KUgNoZjWAe4CjgDnAJDMb6e7TEsk+BXoDVxWiTCKSLmmp0QXV6opI2QrVBNwBmOnuHwGY2VDgRKAkAHT32XHa6gKVSURERCSVCtUEvDPwWWJ4ThwnIiIiIgVWqBrAXG0v69VGYWZ9gD4Au+6664aUSURENmFpadZXk75UhULVAM4BdkkMNwPmrk9G7v6Auxe7e3GTJk0qpXAiIiIiaVKoGsBJQAszaw58DvQCzizQskVERGQjl5YaXdg4anULUgPo7iuBvsDzwHTgSXd/z8z6m1kPADNrb2ZzgFOB+83svUKUTURERCRtCvYiaHcfBYzKGtcv8XkSoWlYRERERKqQfgpOREREJGUUAIqIiIikjAJAERERkZRRACgiIiKSMgoARURERFJGAaCIiIhIyigAFBEREUkZBYAiIiIiKaMAUERERCRlFACKiIiIpIwCQBEREZGUUQAoIiIikjIKAEVERERSRgGgiIiISMooABQRERFJGQWAIiIiIimjAFBEREQkZRQAioiIiKSMAkARERGRlFEAKCIiIpIyCgBFREREUkYBoIiIiEjKKAAUERERSRkFgCIiIiIpowBQREREJGUUAIqIiIikjAJAERERkZRRACgiIiKSMgoARURERFJGAaCIiIhIyigAFBEREUkZBYAiIiIiKaMAUERERCRlFACKiIiIpEzBAkAz625mH5jZTDO7Nsf02mb2RJz+hpkVFapsIiIiImlSkADQzGoA9wDHAPsCZ5jZvlnJzgO+dfc9gTuB/y1E2URERETSplA1gB2Ame7+kbv/CAwFTsxKcyLwcPw8DDjCzKxA5RMRERFJjZoFWs7OwGeJ4TnAgfnSuPtKM1sINAK+TiYysz5Anzi42Mw+qJISb1wak7Udqov1VkxeSbRPNz/ap5sX7c/NT1r26W7lSVSoADDXmvp6pMHdHwAeqIxCbSrMbLK7F1d3OaTy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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7d01612f0e48>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# TODO: Import a supervised learning model that has 'feature_importances_'\n",
    "\n",
    "\n",
    "# TODO: Train the supervised model on the training set using .fit(X_train, y_train)\n",
    "model = RandomForestClassifier(random_state=42)\n",
    "model.fit(X_train, y_train)\n",
    "\n",
    "# TODO: Extract the feature importances using .feature_importances_ \n",
    "importances = model.feature_importances_\n",
    "\n",
    "# Plot\n",
    "vs.feature_plot(importances, X_train, y_train)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Question 7 - Extracting Feature Importance\n",
    "\n",
    "Observe the visualization created above which displays the five most relevant features for predicting if an individual makes at most or above \\$50,000.  \n",
    "* How do these five features compare to the five features you discussed in **Question 6**?\n",
    "* If you were close to the same answer, how does this visualization confirm your thoughts? \n",
    "* If you were not close, why do you think these features are more relevant?"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Answer:**\n",
    "\n",
    "The model’s top features: age, hours per week, capital gain, marital status, and education num, mostly match my original choices. I correctly identified four of them, but I included capital-loss, while the model found marital status more important.\n",
    "\n",
    "This shows that work-related and education factors are strong predictors, which aligns with my reasoning. However, the model highlights that marital status also plays a significant role, likely due to financial stability or dual-income households.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Feature Selection\n",
    "How does a model perform if we only use a subset of all the available features in the data? With less features required to train, the expectation is that training and prediction time is much lower — at the cost of performance metrics. From the visualization above, we see that the top five most important features contribute more than half of the importance of **all** features present in the data. This hints that we can attempt to *reduce the feature space* and simplify the information required for the model to learn. The code cell below will use the same optimized model you found earlier, and train it on the same training set *with only the top five important features*. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Final Model trained on full data\n",
      "------\n",
      "Accuracy on testing data: 0.8619\n",
      "F-score on testing data: 0.7334\n",
      "\n",
      "Final Model trained on reduced data\n",
      "------\n",
      "Accuracy on testing data: 0.8464\n",
      "F-score on testing data: 0.6946\n"
     ]
    }
   ],
   "source": [
    "# Import functionality for cloning a model\n",
    "from sklearn.base import clone\n",
    "\n",
    "# Reduce the feature space\n",
    "X_train_reduced = X_train[X_train.columns.values[(np.argsort(importances)[::-1])[:5]]]\n",
    "X_test_reduced = X_test[X_test.columns.values[(np.argsort(importances)[::-1])[:5]]]\n",
    "\n",
    "# Train on the \"best\" model found from grid search earlier\n",
    "clf = (clone(best_clf)).fit(X_train_reduced, y_train)\n",
    "\n",
    "# Make new predictions\n",
    "reduced_predictions = clf.predict(X_test_reduced)\n",
    "\n",
    "# Report scores from the final model using both versions of data\n",
    "print(\"Final Model trained on full data\\n------\")\n",
    "print(\"Accuracy on testing data: {:.4f}\".format(accuracy_score(y_test, best_predictions)))\n",
    "print(\"F-score on testing data: {:.4f}\".format(fbeta_score(y_test, best_predictions, beta = 0.5)))\n",
    "print(\"\\nFinal Model trained on reduced data\\n------\")\n",
    "print(\"Accuracy on testing data: {:.4f}\".format(accuracy_score(y_test, reduced_predictions)))\n",
    "print(\"F-score on testing data: {:.4f}\".format(fbeta_score(y_test, reduced_predictions, beta = 0.5)))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Question 8 - Effects of Feature Selection\n",
    "\n",
    "* How does the final model's F-score and accuracy score on the reduced data using only five features compare to those same scores when all features are used?\n",
    "* If training time was a factor, would you consider using the reduced data as your training set?"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Answer:**\n",
    "\n",
    "Using only the top five features produced mixed results. The reduced model achieved a slightly higher accuracy score of 0.8464 compared to 0.8619 on the full dataset, but its F-score dropped from 0.7334 to 0.6946. This means the reduced model remained fairly accurate overall, but it became less effective at identifying individuals who earn more than $50,000.\n",
    "\n",
    "These results suggest that the top five features capture much of the important information in the dataset, but the full set of features still provides better predictive performance, especially for the target class of interest. In other words, reducing the feature space improves simplicity and may reduce training complexity, but it comes at the cost of lower model qu"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "> **Note**: Once you have completed all of the code implementations and successfully answered each question above, you may finalize your work by exporting the iPython Notebook as an HTML document. You can do this by using the menu above and navigating to  \n",
    "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Before You Submit\n",
    "You will also need run the following in order to convert the Jupyter notebook into HTML, so that your submission will include both files."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "!!jupyter nbconvert *.ipynb"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 1
}
