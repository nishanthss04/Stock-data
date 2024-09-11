from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, lit
from pyspark.sql.types import TimestampType, FloatType
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import LogisticRegression
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

spark = SparkSession.builder \
    .appName("BigDataLSTM") \
    .getOrCreate()

stock_data = spark.read.csv(r"C:\\Users\\Nishanth\\.vscode\\Python\\AAPL_stock_data.csv", header=True, inferSchema=True)
stock_data = stock_data.withColumn('Datetime', col('Datetime').cast(TimestampType()))
current_time = stock_data.agg({'Datetime': 'max'}).collect()[0][0]
five_hours_ago = current_time - timedelta(hours=5)
recent_stock_data = stock_data.filter(col('Datetime') >= lit(five_hours_ago))

if recent_stock_data.count() == 0:
    raise ValueError("No stock data available for the last 5 hours.")
sentiment_data = spark.read.csv(r"C:\\Users\\Nishanth\\.vscode\\Python\\sentiment_analysis_results.csv", header=True, inferSchema=True)
average_sentiment_score = sentiment_data.agg(avg('Score')).collect()[0][0]
recent_stock_data = recent_stock_data.withColumn('Sentiment_Score', lit(average_sentiment_score))
feature_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Sentiment_Score']
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
assembled_data = assembler.transform(recent_stock_data)

scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=True)
scaler_model = scaler.fit(assembled_data)
scaled_data = scaler_model.transform(assembled_data)

pandas_df = scaled_data.select("scaled_features").toPandas()
X = np.array(pandas_df['scaled_features'].tolist())
X = X.reshape((X.shape[0], 1, X.shape[1]))

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam

model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid')) 

model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, np.ones(X.shape[0]), epochs=10, batch_size=32) 

predictions = model.predict(X)
threshold = 0.4
buy_signal = predictions > threshold

results = pd.DataFrame({
    'prediction_value': predictions.flatten(),  
    'prediction_class': buy_signal.flatten()     
})


results_spark_df = spark.createDataFrame(results)

print(f"Prediction: {predictions}, Buy Signal: {buy_signal}")

predicted_classes = (predictions > threshold).astype(int)

predicted_df = pd.DataFrame(predicted_classes, columns=['Buy_Signal'])
spark_df = spark.createDataFrame(predicted_df)

output_path = r"C:\\Users\\Nishanth\\.vscode\\Python\\output.csv" 
results_spark_df.write.csv(output_path, header=True, mode="overwrite")

spark.stop()