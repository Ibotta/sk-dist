python_version=$(python -c 'import sys; print(sys.version_info[:][1] < 8)')

# only run for python version under 3.8
if [ $python_version == "True" ]
then
  # install spark
  PATH=$(echo "$PATH" | sed -e 's/:\/usr\/local\/lib\/jvm\/openjdk11\/bin//')
  JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk-amd64
  mkdir -p /opt
  wget -q -O /opt/spark.tgz https://mirrors.ocf.berkeley.edu/apache/spark/spark-2.4.7/spark-2.4.7-bin-hadoop2.7.tgz
  tar xzf /opt/spark.tgz -C /opt/
  rm /opt/spark.tgz
  export SPARK_HOME=/opt/spark-2.4.7-bin-hadoop2.7
  export PATH=$PATH:/opt/spark-2.4.7-bin-hadoop2.7/bin

  # run tests
  pip install -e .[tests]
  build_tools/test_script.sh
else
  echo "Skipped tests - spark and python versions are not compatible"
fi
