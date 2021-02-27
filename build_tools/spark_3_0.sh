# install spark
PATH=$(echo "$PATH" | sed -e 's/:\/usr\/local\/lib\/jvm\/openjdk11\/bin//')
JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk-amd64
mkdir -p /opt
wget -q -O /opt/spark.tgz https://mirrors.gigenet.com/apache/spark/spark-3.0.2/spark-3.0.2-bin-hadoop2.7.tgz
tar xzf /opt/spark.tgz -C /opt/
rm /opt/spark.tgz
export SPARK_HOME=/opt/spark-3.0.2-bin-hadoop2.7
export PATH=$PATH:/opt/spark-3.0.2-bin-hadoop2.7/bin

# run tests
pip install -e .[tests]
build_tools/test_script.sh
