#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
df = pd.read_csv('Heartrate.csv')

df


# In[2]:


from datetime import datetime

# Convert 'date' column to datetime format with specified format
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')

# Extract day of the week and month name into new columns
df['Day'] = df['Date'].dt.strftime('%A') # Full name of the day (e.g., Monday)
df['Month'] = df['Date'].dt.strftime('%B') # Full name of the month (e.g., January)

print(df)


# In[3]:


#dropping unwanted columns
columns_to_drop = ['Id','Date']
df= df.drop(columns_to_drop,axis = 1)


# In[4]:


df


# In[5]:


corrmatrix = df.corr()
plt.subplots(figsize = (20,8))
sns.heatmap(corrmatrix,vmin = -0.7,vmax= 0.7, annot = True,linewidths=0.2,cmap='YlGnBu')


# In[6]:


df.isnull().sum()


# In[7]:


from sklearn.preprocessing import LabelEncoder

# Initialize LabelEncoder for 'day' column
day_encoder = LabelEncoder()
df['Day_encoded'] = day_encoder.fit_transform(df['Day'])

# Initialize LabelEncoder for 'month' column
month_encoder = LabelEncoder()
df['Month_encoded'] = month_encoder.fit_transform(df['Month'])

df


# In[8]:


#dropping unwanted columns
columns_to_drop = ['Day','Month']
df= df.drop(columns_to_drop,axis = 1)
df


# In[9]:


sns.boxplot(df)
plt.title('Box Plot for Outlier Detection')
plt.show()


# In[10]:


# treating the outlier

q1 = df['TotalSteps'].quantile (0.25)
q3 = df['TotalSteps'].quantile (0.75)

iqr = q3-q1
iqr


# In[11]:


upper_limit = q3+1.5*iqr
lower_limit = q1-1.5*iqr

print(upper_limit)
print(lower_limit)


# In[12]:


def limit_imputer(value):
    if value>upper_limit:
        return upper_limit
    if value<lower_limit:
        return lower_limit
    else:
        return value


# In[13]:


#applying the imputer function

df['TotalSteps']= df['TotalSteps'].apply(limit_imputer)


# In[14]:


sns.boxplot(df)
plt.title('Box Plot for Outlier Detection')
plt.show()


# In[15]:


sns.boxplot(x = df ['TotalActiveMinutes'])


# In[16]:


q1 = df['TotalActiveMinutes'].quantile (0.25)
q3 = df['TotalActiveMinutes'].quantile (0.75)

iqr = q3-q1
iqr


# In[17]:


upper_limit = q3+1.5*iqr
lower_limit = q1-1.5*iqr

print(upper_limit)
print(lower_limit)


# In[18]:


df['TotalActiveMinutes']= df['TotalActiveMinutes'].apply(limit_imputer)


# In[19]:


sns.boxplot(x = df ['TotalDistance'])


# In[20]:


q1 = df['TotalDistance'].quantile (0.25)
q3 = df['TotalDistance'].quantile (0.75)

iqr = q3-q1
iqr


# In[21]:


upper_limit = q3+1.5*iqr
lower_limit = q1-1.5*iqr

print(upper_limit)
print(lower_limit)


# In[22]:


df['TotalDistance']= df['TotalDistance'].apply(limit_imputer)


# In[23]:


sns.boxplot(df)
plt.title('Box Plot for Outlier Detection')
plt.show()


# In[24]:


sns.boxplot(x = df ['Heart_rate'])


# In[25]:


q1 = df['Heart_rate'].quantile (0.25)
q3 = df['Heart_rate'].quantile (0.75)

iqr = q3-q1
iqr


# In[26]:


upper_limit = q3+1.5*iqr
lower_limit = q1-1.5*iqr

print(upper_limit)
print(lower_limit)


# In[27]:


df['Heart_rate']= df['Heart_rate'].apply(limit_imputer)


# In[28]:


sns.boxplot(df)
plt.title('Box Plot for Outlier Detection')
plt.show()


# In[29]:


x_df = df.drop('Calories', axis = 1)
y_df = df['Calories']


# In[30]:


x_df


# In[31]:


y_df


# In[32]:


from sklearn.model_selection import train_test_split


# In[33]:


x_train, x_test, y_train, y_test = train_test_split(x_df,y_df,test_size=0.05,random_state=46)


# ### The model gave more accurate values without scaling the data

# In[34]:


x_train.shape


# In[35]:


y_train.shape


# # Linear Regression

# In[36]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# In[37]:


linear_regressor = LinearRegression()
linear_regressor.fit(x_train, y_train)


# In[38]:


#making predictions
y_pred = linear_regressor.predict(x_test)


# In[39]:


# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
print("Coefficients:", linear_regressor.coef_)
print("Intercept:", linear_regressor.intercept_)


# In[40]:


from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)
print("R² Score:", r2)


# # Ridge Regression

# In[41]:


from sklearn.linear_model import Ridge

# Creating and training the Ridge Regression model
ridge_regressor = Ridge(alpha=1.0)
ridge_regressor.fit(x_train, y_train)

# Making predictions
y_pred_ridge = ridge_regressor.predict(x_test)

# Evaluating the model
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
print("Mean Squared Error (Ridge):", mse_ridge)
print("Coefficients (Ridge):", ridge_regressor.coef_)
print("Intercept (Ridge):", ridge_regressor.intercept_)


# In[42]:


r2r = r2_score(y_test, y_pred_ridge)
print("R² Score:", r2r)


# In[43]:


from sklearn.linear_model import Lasso

# Creating and training the Lasso Regression model
lasso_regressor = Lasso(alpha=0.1)
lasso_regressor.fit(x_train, y_train)

# Making predictions
y_pred_lasso = lasso_regressor.predict(x_test)

# Evaluating the model
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
print("Mean Squared Error (Lasso):", mse_lasso)
print("Coefficients (Lasso):", lasso_regressor.coef_)
print("Intercept (Lasso):", lasso_regressor.intercept_)


# In[44]:


r2l = r2_score(y_test, y_pred_lasso)
print("R² Score:", r2l)


# # Polynomial Regression

# In[45]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Generating polynomial features
poly_features = PolynomialFeatures(degree=2)

# Creating and training the Polynomial Regression model
poly_regressor = make_pipeline(poly_features, Ridge(alpha=0.1))
poly_regressor.fit(x_train, y_train)

# Making predictions
y_pred_poly = poly_regressor.predict(x_test)

# Evaluating the model
mse_poly = mean_squared_error(y_test, y_pred_poly)
print("Mean Squared Error (Polynomial):", mse_poly)


# In[46]:


r2p = r2_score(y_test, y_pred_poly)
print("R² Score:", r2p)


# # Comparing Models

# In[47]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score


# In[48]:


models = {
    'Linear Regression': LinearRegression(),
    'Polynomial Regression (degree=2)': make_pipeline(PolynomialFeatures(degree=2), LinearRegression()),
    'Ridge Regression': Ridge(alpha=0.1),
    'Lasso Regression': Lasso(alpha=0.1),
    'Support Vector Regression': SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

# Evaluate models
results = {}

for name, model in models.items():
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {'MSE': mse, 'R²': r2}

# Print results
for name, metrics in results.items():
    print(f"{name}: MSE = {metrics['MSE']:.4f}, R² = {metrics['R²']:.4f}")


# In[49]:


import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 3. Set up the pipeline
pipeline = Pipeline([
    ('poly_features', PolynomialFeatures()),
    ('scaler', StandardScaler()),
    ('regressor', Ridge())  # You can use LinearRegression() instead of Ridge() if you don't want regularization
])

# 4. Define the parameter grid
param_grid = {
    'poly_features__degree': [2, 3, 4, 5],
    'regressor__alpha': [0.01, 0.1, 1, 10, 100]  # Only applicable if you use Ridge
}

# 5. Use GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(x_train, y_train)

# 6. Evaluate the results
print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation score (negative MSE): ", grid_search.best_score_)

# Predictions and evaluation on the test set
y_pred = grid_search.best_estimator_.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Test set Mean Squared Error: ", mse)
print("Test set R² Score: ", r2)


# In[50]:


rf = RandomForestRegressor()

# Define the parameter grid
param_grid = {
    'n_estimators': [100],
    'max_depth': [20],
    'min_samples_split': [10],
    'min_samples_leaf': [1],
    'bootstrap': [True]
}

# Set up the GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

# Fit the grid search to the data
grid_search.fit(x_train, y_train)

# Print the best parameters and the best score
print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation score (negative MSE): ", grid_search.best_score_)

# Evaluate the best model on the test set
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
print("Test set Mean Squared Error: ", mse)
r2 = r2_score(y_test, y_pred)
print("Test set R² Score: ", r2)


# In[51]:


# Random Forest
plt.subplot(1, 5, 2)
plt.scatter(y_test, y_pred, color='green', edgecolors='k')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2)
plt.xlabel('Actual Calories')
plt.ylabel('Predicted Calories')
plt.title('Random Forest Regressor')


# In[52]:


X_split = np.array([[5135, 3.39, 318, 76.639377,3,2]])

# Make predictions for the input values
calories_predictions = best_rf.predict(X_split)

# Print the predictions
print("Predicted Calories:", calories_predictions)


# In[53]:


# Example input values (replace with your own input values)
X_split = np.array([[5135, 3.39, 318, 76.639377,3,2]]) 
# Make predictions for the input values
calories_predictions = poly_regressor.predict(X_split)

# Print the predictions
print("Predicted Calories:", calories_predictions)


# In[ ]:




