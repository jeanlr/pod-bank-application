FROM openjdk:11-slim

ENV SPARK_VERSION=3.5.6
ENV HADOOP_VERSION=3
ENV PYSPARK_PYTHON=python3
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:/opt/spark/bin:${PATH}"

RUN apt-get update && \
    apt-get install -y wget python3 python3-pip openjdk-11-jdk && \
    apt-get clean

RUN groupadd -g 1000 jovyan && \
    useradd -m -u 1000 -g jovyan jovyan

COPY requirements.txt /tmp/requirements.txt

RUN pip3 install --no-cache-dir jupyter pyspark findspark && \
    pip3 install --no-cache-dir -r /tmp/requirements.txt

RUN wget https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz && \
    tar -xzf spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz -C /opt && \
    mv /opt/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} /opt/spark && \
    rm spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz

ENV SPARK_HOME=/opt/spark

WORKDIR /home/jovyan

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

USER root

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--ServerApp.token=''"]
