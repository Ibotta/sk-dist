python_version=$(python -c 'import sys; print(sys.version_info[:][1] < 8)')

# only run for python version under 3.8
if [ $python_version == "True" ]
then
  # install spark
  PATH=$(echo "$PATH" | sed -e 's/:\/usr\/local\/lib\/jvm\/openjdk11\/bin//')
  JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk-amd64
  mkdir -p /opt
  echo "downloading spark"
  wget -q -O /opt/spark.tgz https://www.apache.org/dyn/closer.lua/spark/spark-2.4.8/spark-2.4.8-bin-hadoop2.7.tgz
  echo "before untar"
  tar xzvf /opt/spark.tgz -C /opt/
  echo "after untar"
  rm /opt/spark.tgz
  export SPARK_HOME=/opt/spark-2.4.8-bin-hadoop2.7
  export PATH=$PATH:/opt/spark-2.4.8-bin-hadoop2.7/bin

  # run tests
  pip install -e .[tests]
  build_tools/test_script.sh
else
  echo "Skipped tests - spark and python versions are not compatible"
fi
