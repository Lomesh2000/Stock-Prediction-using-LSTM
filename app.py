import os
import glob
import shutil

#os.chdir('/content/drive/MyDrive/Github')
#os.mkdir('Stock prediction')
#os.chdir('/content/drive/MyDrive/Github/Stock prediction')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import pandas_datareader as data

from keras.models import load_model
import streamlit as st

#######################

start='2010-01-01'
end='2022-01-01'

st.title('Stock Trend Prediction')

user_input=st.text_input('Enter Stock Ticker','AAPL')
df=data.DataReader(user_input,'yahoo',start,end)

st.subheader("Data from 2010 - 2022")
st.write(df.describe())

#visualing Data
st.subheader('Closing Price vs Time')
figure=plt.figure(figsize=((12,6)))
plt.plot(df.Close)
st.pyplot(figure)


st.subheader('Closing Price vs Time chart of Mean Average of previous hundreds')
ma_100=df.Close.rolling(100).mean()
figure=plt.figure(figsize=((12,6)))
plt.plot(ma_100)
plt.plot(df.Close)
st.pyplot(figure)


st.subheader('Closing Price vs Time chart of Mean average of previus 100 & Mean average of previous 200 ')
ma_100=df.Close.rolling(100).mean()
ma_200=df.Close.rolling(200).mean()
figure=plt.figure(figsize=((12,6)))
plt.plot(ma_100)
plt.plot(ma_200)
plt.plot(df.Close)
st.pyplot(figure)

#splitting into train and test data
train_data=pd.DataFrame(df['Close'][0:int(len(df['Close'])*0.7)])
test_data=pd.DataFrame(df['Close'][len(train_data)::])

#Scaling data
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))

#transforming
train_data_array=scaler.fit_transform(train_data)

#spltting into x_train and y_train
x_train=[]
y_train=[]

for i in range(100,len(train_data_array)):
    x_train.append(train_data_array[i-100:i])
    y_train.append(train_data_array[i])

#convering into arrays
x_train=np.array(x_train)
y_train=np.array(y_train)

#load model
model=load_model('stock and prediction model.h5')


#taking past 100 days data
previous_100_days=train_data.tail(100)

final_data=previous_100_days.append(test_data,ignore_index=True)


#transforming final_data
input_data=scaler.fit_transform(final_data)


#splitting into x_test and y_test
x_test=[] #For predicition	
y_test=[] #To compare with the predictions

for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i])
#converting into arrays
x_test=np.array(x_test)
y_test=np.array(y_test)
#x_test.shape,y_test.shape

#predicting future prices using 
y_predicted=model.predict(x_test)

#Scaling up 
scale_factor=scaler.scale_[0]

y_test=y_test*(1/scale_factor)

y_predicted=y_predicted*(1/scale_factor)

#Final Graph
st.subheader("Predicted Prices vs Original Prices")
fig2=plt.figure(figsize=(12,9))
plt.plot(y_test,'r',label="Original Price")

plt.plot(y_predicted,'g',label="Predicted Price")

plt.xlabel('Time')
plt.ylabel('Price')

plt.legend()
plt.savefig('original vs predicted prices.png')
st.pyplot(fig2)



