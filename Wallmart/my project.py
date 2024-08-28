#!/usr/bin/env python
# coding: utf-8

# In[32]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm


# In[4]:


data = pd.read_csv('Walmart DataSet.csv')


# In[5]:


data


# In[57]:


data.isnull().sum()


# In[6]:


data.info()


# In[7]:


data['Date'] = pd.to_datetime(data['Date'], dayfirst= True)


# In[8]:


data


# In[9]:


data.info()


# In[10]:


data['year'] = pd.DatetimeIndex(data['Date']).year


# In[11]:


data


# In[13]:


store_sales= data.groupby(['Store' , 'year'])[['Weekly_Sales']].sum().unstack()
store_sales


# In[14]:


from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px


# In[15]:


fig = make_subplots(rows=3, cols=1)

fig.append_trace(go.Scatter(x=store_sales.index,y = store_sales.iloc[:,0], name="2010"), row=1, col=1)

fig.append_trace(go.Scatter(x=store_sales.index,y = store_sales.iloc[:,1],name="2011"), row=2, col=1)

fig.append_trace(go.Scatter(x=store_sales.index,y = store_sales.iloc[:,2],name="2012"), row=3, col=1)


fig.update_layout(height=600, width=1000, title_text="Total Sales in each Year")
fig.show()


# In[16]:


plt.figure(figsize = (20,5))
fig = px.scatter(data, x="Weekly_Sales", y="Temperature", color="Store",
                 title="Relation between Temperature and weeklysales within stores")

fig.show()


# In[58]:


plt.figure(figsize = (50,5))
fig = px.scatter(data, x="Weekly_Sales", y="CPI", color="Store",
                 title="Relation between CPI and weeklysales within stores")

fig.show()


# In[18]:


plt.figure(figsize = (20,5))
fig = px.scatter(data, x="Weekly_Sales", y="Unemployment", color="Store",
                 title="Relation between Unemployment and weeklysales within stores" , color_continuous_scale=px.colors.sequential.Viridis)

fig.show()


# In[19]:


plt.figure(figsize=(15,5))
sns.heatmap(data.corr(), annot=True)


# In[21]:


data.set_index('Date', inplace=True)
a= int(input("Enter the store id:"))
store = data[data.Store == a]
sales = pd.DataFrame(store.Weekly_Sales.groupby(store.index).sum())
sales.dtypes


# In[22]:


sales


# In[23]:


fig = px.line(sales, y="Weekly_Sales")
fig.show()


# In[39]:


decomposition = sm.tsa.seasonal_decompose (sales.Weekly_Sales, period=12)
fig = decomposition.plot()  
fig.set_size_inches(12, 10)
plt.show()


# In[40]:


mean_log = sales.rolling(window=12).mean()  
std_log = sales.rolling(window=12).std()    

plt.plot(sales, color='blue', label='Original')
plt.plot(mean_log, color='red', label='Rolling Mean')
plt.plot(std_log, color='black', label='Rolling Std')
plt.legend(loc='best')
plt.title('Rolling Mean & Standard Deviation (Logarithmic Scale)')


# In[38]:


from statsmodels.tsa.stattools import adfuller
result = adfuller(sales['Weekly_Sales'])
result


# In[25]:


p_value=result[1]
p_value


# In[26]:


if p_value <=0.05:
    print('Stationarity is present')
else:
    print('NO Stationarity is present')


# In[27]:


from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import pacf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
acf_plot=acf(sales)
pacf_plot=pacf(sales)
plot_acf(acf_plot);


# In[42]:


from pmdarima import auto_arima
model = auto_arima(sales, seasonal=True, stepwise=True, trace=True)


# In[43]:


from statsmodels.tsa.arima.model import ARIMA  

train = sales.iloc[:120]['Weekly_Sales']
test =  sales.iloc[121:]['Weekly_Sales']
model = ARIMA(train, order=(1,1,1)) 
model_fit = model.fit()
model_fit.summary()


# In[44]:


fc= model_fit.forecast(steps=24, alpha=0.05)
fc_series = pd.Series(fc)
plt.figure(figsize=(12,5), dpi=100)
plt.plot(train, label='training')
plt.plot(test, label='actual')
plt.plot(fc_series, label='forecast')
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()


# In[24]:


b= int(input("Enter the store id:"))
store = data[data.Store == b]
sales1 = pd.DataFrame(store.Weekly_Sales.groupby(store.index).sum())
sales1.dtypes


# In[29]:


sales1


# In[30]:


fig = px.line(sales1, y="Weekly_Sales")
fig.show()


# In[36]:


decomposition = sm.tsa.seasonal_decompose (sales1.Weekly_Sales, period=12)
fig = decomposition.plot()  
fig.set_size_inches(12, 10)
plt.show()


# In[45]:


result1 = adfuller(sales1['Weekly_Sales'])
result1
p_value=result1[1]
p_value


# In[46]:


if p_value <=0.05:
    print('Stationarity is present')
else:
    print('NO Stationarity is present')


# In[39]:


acf_plot=acf(sales1)
pacf_plot=pacf(sales1)
plot_acf(acf_plot);


# In[47]:


model = auto_arima(sales1, seasonal=True, stepwise=True, trace=True)


# In[49]:


fc= model_fit.forecast(steps=24, alpha=0.05)
fc_series = pd.Series(fc)
plt.figure(figsize=(12,5), dpi=100)
plt.plot(train, label='training')
plt.plot(test, label='actual')
plt.plot(fc_series, label='forecast')
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()


# In[48]:


train = sales1.iloc[:120]['Weekly_Sales']
test =  sales1.iloc[121:]['Weekly_Sales']
model= ARIMA(train, order=(2,1,3)) 
model_fit = model.fit()
model_fit.summary()


# In[60]:


c= int(input("Enter the store id:"))
store = data[data.Store == c]
sales2 = pd.DataFrame(store.Weekly_Sales.groupby(store.index).sum())
sales2.dtypes


# In[61]:


sales2


# In[62]:


fig = px.line(sales2, y="Weekly_Sales")
fig.show()


# In[52]:


decomposition = sm.tsa.seasonal_decompose (sales2.Weekly_Sales, period=12)
fig = decomposition.plot()  
fig.set_size_inches(12, 10)
plt.show()


# In[53]:


result2 = adfuller(sales2['Weekly_Sales'])
result2
p_value=result2[1]
p_value


# In[54]:


if p_value <=0.05:
    print('Stationarity is present')
else:
    print('NO Stationarity is present')


# In[55]:


acf_plot=acf(sales2)
pacf_plot=pacf(sales2)
plot_acf(acf_plot);


# In[56]:


model = auto_arima(sales2, seasonal=True, stepwise=True, trace=True)


# In[57]:


train = sales2.iloc[:120]['Weekly_Sales']
test =  sales2.iloc[121:]['Weekly_Sales']
model= ARIMA(train, order=(1,0,0)) 
model_fit = model.fit()
model_fit.summary()


# In[58]:


fc= model_fit.forecast(steps=24, alpha=0.05)
fc_series = pd.Series(fc)
plt.figure(figsize=(12,5), dpi=100)
plt.plot(train, label='training')
plt.plot(test, label='actual')
plt.plot(fc_series, label='forecast')
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()


# In[52]:


d= int(input("Enter the store id:"))
store = data[data.Store == d]
sales3 = pd.DataFrame(store.Weekly_Sales.groupby(store.index).sum())
sales3.dtypes


# In[53]:


sales3


# In[54]:


fig = px.line(sales3, y="Weekly_Sales")
fig.show()


# In[62]:


decomposition = sm.tsa.seasonal_decompose (sales3.Weekly_Sales, period=12)
fig = decomposition.plot()  
fig.set_size_inches(12, 10)
plt.show()


# In[66]:


result3 = adfuller(sales3['Weekly_Sales'])
result3
p_value=result3[1]
p_value


# In[67]:


if p_value <=0.05:
    print('Stationarity is present')
else:
    print('NO Stationarity is present')


# In[68]:


acf_plot=acf(sales3)
pacf_plot=pacf(sales3)
plot_acf(acf_plot);


# In[69]:


model = auto_arima(sales3, seasonal=True, stepwise=True, trace=True)


# In[70]:


train = sales3.iloc[:120]['Weekly_Sales']
test =  sales3.iloc[121:]['Weekly_Sales']
model= ARIMA(train, order=(1,0,0))
model_fit = model.fit()
model_fit.summary()


# In[71]:


fc= model_fit.forecast(steps=24, alpha=0.05)
fc_series = pd.Series(fc)
plt.figure(figsize=(12,5), dpi=100)
plt.plot(train, label='training')
plt.plot(test, label='actual')
plt.plot(fc_series, label='forecast')
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()


# In[55]:


e= int(input("Enter the store id:"))
store = data[data.Store == e]
sales4 = pd.DataFrame(store.Weekly_Sales.groupby(store.index).sum())
sales4.dtypes


# In[73]:


sales4


# In[56]:


fig = px.line(sales4, y="Weekly_Sales")
fig.show()


# In[75]:


decomposition = sm.tsa.seasonal_decompose (sales4.Weekly_Sales, period=12)
fig = decomposition.plot()  
fig.set_size_inches(12, 10)
plt.show()


# In[76]:


result4 = adfuller(sales4['Weekly_Sales'])
result4
p_value=result4[1]
p_value


# In[77]:


if p_value <=0.05:
    print('Stationarity is present')
else:
    print('NO Stationarity is present')


# In[78]:


acf_plot=acf(sales4)
pacf_plot=pacf(sales4)
plot_acf(acf_plot);


# In[79]:


model = auto_arima(sales4, seasonal=True, stepwise=True, trace=True)


# In[80]:


train = sales4.iloc[:120]['Weekly_Sales']
test =  sales4.iloc[121:]['Weekly_Sales']
model= ARIMA(train, order=(1,0,0)) 
model_fit = model.fit()
model_fit.summary()


# In[81]:


fc= model_fit.forecast(steps=24, alpha=0.05)
fc_series = pd.Series(fc)
plt.figure(figsize=(12,5), dpi=100)
plt.plot(train, label='training')
plt.plot(test, label='actual')
plt.plot(fc_series, label='forecast')
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()


# In[ ]:




