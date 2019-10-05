PATH=$(echo "$PATH" | sed -e 's/:\/usr\/local\/lib\/jvm\/openjdk11\/bin//')
JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk-amd64
mkdir -p /opt
wget -q -O /opt/spark.tgz http://www.gtlib.gatech.edu/pub/apache/spark/spark-2.4.4/spark-2.4.4-bin-hadoop2.7.tgz
tar xzf /opt/spark.tgz -C /opt/
rm /opt/spark.tgz
export SPARK_HOME=/opt/spark-2.4.4-bin-hadoop2.7
export PATH=$PATH:/opt/spark-2.4.4-bin-hadoop2.7/bin
echo $SPARK_HOME
echo $PATH
