from pyspark import SparkContext                                                                                        
from pyspark.sql import SparkSession                                                                                    
from pyspark.streaming import StreamingContext                                                                          
from pyspark.streaming.kafka import KafkaUtils 
import json
from common_data_processor import predict 


app_name = "Housing Price Prediction"
sc = SparkContext(appName=app_name) 
ssc = StreamingContext(sc, 5)
spark = SparkSession.builder.appName(app_name).config('spark.ui.port', '4050').getOrCreate()

dks = KafkaUtils.createDirectStream(ssc, topics=["housing-price-test"], kafkaParams={"metadata.broker.list":"localhost:9092"})

                                                                                                                
houde_details = dks.map(lambda x: json.dumps(x[1]))

# call predict
predict(houde_details)

ssc.start()                                                                                                             
ssc.awaitTermination()