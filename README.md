# Simple House Price Prediction Using Linear Regression

## Project Overview

This project builds a **Simple Linear Regression** model to estimate
house prices using the living area of a house (`sqft_living`). It is
based on the house-price dataset used in the accompanying Google Colab
notebook.

The project follows a basic supervised-learning workflow: data
inspection, exploratory data analysis (EDA), feature selection,
train-test split, model training, evaluation, visualization, and
prediction for a new example.

## Objective

Predict the house price (`price`) using one numerical input feature:

-   **Input feature (X):** `sqft_living` --- living area of the house in
    square feet.
-   **Target (y):** `price` --- house price.

The notebook compares numerical feature correlations with `price` and
selects `sqft_living`, which has the highest positive correlation among
the listed numerical input features in this dataset (approximately
**0.4304**).

## Dataset

The notebook loads a CSV file named `data.csv`.

-   Number of rows: **4,600**
-   Number of columns: **18**
-   Missing values: **None detected** by `df.isnull().sum()`
-   Target column: `price`
-   Selected feature: `sqft_living`

The dataset includes numerical and categorical columns, such as
bedrooms, bathrooms, living area, lot area, city, and country. This
project uses only `sqft_living` for the simple regression model.

> Place `data.csv` in the location expected by the notebook. The current
> notebook uses `/content/data.csv`, which is a Google Colab path.
> Change the path if you run the notebook in a different environment.

## Tools and Libraries

-   Python
-   Pandas
-   NumPy
-   Matplotlib
-   Seaborn
-   Scikit-learn
-   Google Colab / Jupyter Notebook

## Workflow

1.  **Import libraries** --- load the Python libraries used for data
    handling, visualization, and machine learning.
2.  **Load and inspect the dataset** --- read `data.csv`, inspect sample
    rows and dimensions, check missing values, and review data types.
3.  **Correlation analysis** --- calculate correlations between
    numerical columns and the target `price`.
4.  **Feature selection** --- select `sqft_living` as the single input
    feature and `price` as the target.
5.  **Exploratory data analysis** --- plot the price distribution,
    inspect the price boxplot, and visualize living area against house
    price.
6.  **Split the data** --- use an 80:20 train-test split with
    `random_state=42`.
7.  **Scale the feature** --- fit `StandardScaler` on the training
    feature and use it to transform both training and test features.
8.  **Train the model** --- fit `LinearRegression` on the scaled
    training feature.
9.  **Evaluate the model** --- calculate MAE, MSE, RMSE, and R² on the
    test set.
10. **Visualize and predict** --- plot the test observations and
    regression predictions, then demonstrate a prediction for a new
    house.

## Model Results

The notebook reports the following test-set metrics:

  Metric                                         Result
  -------------------------------- --------------------
  Mean Absolute Error (MAE)                  225,375.25
  Mean Squared Error (MSE)           990,204,087,727.14
  Root Mean Squared Error (RMSE)             995,089.99
  R² score                                       0.0291

The fitted model's reported parameters, when using the standardized
feature, are:

-   Coefficient: **256,157.89**
-   Intercept: **544,848.27**

Because the input feature is standardized before fitting, this
coefficient represents the estimated change in predicted price for a
**one-standard-deviation increase** in living area. It is not the price
change per additional square foot.

### Interpretation

The R² score of approximately **0.029** indicates that this one-feature
linear model explains only a small proportion of the variation in house
prices in the test data. The errors are also large. Therefore, the model
has weak predictive performance and `sqft_living` alone is not
sufficient to explain house prices accurately.

This is an honest result of the experiment; the notebook evaluates the
model rather than claiming that it predicts prices accurately.

## Predicting a New House

The notebook's model is trained on **scaled** input values. Therefore, a
new raw living-area value must be transformed with the **same fitted
scaler** before prediction.

Use this corrected prediction code after the notebook has fitted
`scaler` and `model1`:

``` python
new_house = pd.DataFrame({
    "sqft_living": [2000]
})

new_house_scaled = scaler.transform(new_house)
predicted_price = model1.predict(new_house_scaled)

print("Predicted House Price:", predicted_price[0])
```

Do not pass the raw value `2000` directly to a model trained on
standardized values. Doing so mixes two different feature scales and can
produce an invalid prediction. The warning in the notebook
(`X has feature names, but LinearRegression was fitted without feature names`)
also occurs because the model was fitted using the NumPy array returned
by `StandardScaler`, but prediction was attempted with a DataFrame.

## How to Run

1.  Open `ML_cie2.ipynb` in Google Colab or Jupyter Notebook.
2.  Upload `data.csv` to the runtime, or update the CSV path in the
    data-loading cell.
3.  Run the notebook cells from top to bottom.
4.  For the new-house example, use the corrected scaled-input prediction
    code above.

## Conclusion

This project demonstrates a complete introductory Simple Linear
Regression workflow for house-price prediction. `sqft_living` was
selected based on its correlation with price, and the model was trained
and evaluated using a held-out test set. The low R² score shows that a
single feature has limited predictive power for this dataset. A future
experiment could investigate multiple relevant features and compare its
performance with this simple baseline.

------------------------------------------------------------------------

**Project notebook:** `ML_cie2.ipynb`\
**Dataset file expected by the notebook:** `data.csv`
