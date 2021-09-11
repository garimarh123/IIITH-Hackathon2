FROM ubuntu:18.04

RUN apt-get -y update
RUN apt-get -y install openjdk-8-jdk

RUN wget http://apache.claz.org/kafka/2.2.0/kafka_2.12-2.2.0.tgz
RUN tar -xvf kafka_2.12-2.2.0.tgz
RUN mv kafka_2.12-2.2.0.tgz kafka

RUN sudo apt install openjdk-8-jdk -y
RUN java -version


ENV USER group12
ENV HADOOP_HOME /home/$USER/hadoop
ENV HADOOP_CONF_DIR $HADOOP_HOME/etc/hadoop
ENV HADOOP_HDFS_HOME $HADOOP_HOME
ENV HADOOP_INSTALL $HADOOP_HOME
ENV HADOOP_MAPRED_HOME $HADOOP_HOME
ENV HADOOP_COMMON_HOME $HADOOP_HOME
ENV HADOOP_HDFS_HOME $HADOOP_HOME
ENV YARN_HOME $HADOOP_HOME
ENV HADOOP_COMMON_LIB_NATIVE_DIR $HADOOP_HOME/lib/native
ENV PATH $PATH:$HADOOP_HOME/sbin:$HADOOP_HOME/bin
ENV HADOOP_OPTS "-Djava.library.path=$HADOOP_HOME/lib/native"
ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64


RUN wget https://archive.apache.org/dist/hadoop/common/hadoop-2.8.5/hadoop-2.8.5.tar.gz
RUN tar -xvf hadoop-2.8.5.tar.gz
RUN mv hadoop-2.8.5.tar.gz hadoop

RUN touch $HADOOP_HOME/etc/hadoop/hadoop-env.sh
RUN cat config/hadoop-env.sh >> $HADOOP_HOME/etc/hadoop/hadoop-env.sh

RUN touch $HADOOP_HOME/etc/hadoop/core-site.xml
RUN cat config/core-site.xml >> $HADOOP_HOME/etc/hadoop/core-site.xml

RUN touch $HADOOP_HOME/etc/hadoop/hdfs-site.xml
RUN cat config/hdfs-site.xml >> $HADOOP_HOME/etc/hadoop/hdfs-site.xml


RUN tar -xvf Downloads/spark-2.4.3-bin-hadoop2.7.tgz
RUN mv spark-2.4.3-bin-hadoop2.7.tgz spark
RUN sudo apt install scala -y
RUN scala -version
ENV PATH $PATH:/home/$USER/spark/bin
ENV PYSPARK_PYTHON  python3


RUN ./kafka_2.13-2.7.0/bin/zookeeper-server-start.sh -daemon ./kafka_2.13-2.7.0/config/zookeeper.properties
RUN ./kafka_2.13-2.7.0/bin/kafka-server-start.sh -daemon ./kafka_2.13-2.7.0/config/server.properties

RUN ./kafka_2.13-2.7.0/bin/kafka-topics.sh --create --bootstrap-server 127.0.0.1:9092 --replication-factor 1 --partitions 2 --topic housing-price-test

RUN ./kafka_2.13-2.7.0/bin/kafka-topics.sh --describe --bootstrap-server 127.0.0.1:9092 --topic housing-price-train

RUN python3 common_data_processor.py