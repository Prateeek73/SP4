FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip openjdk-11-jdk supervisor curl wget tar gzip && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3.10 /usr/bin/python
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV PATH=$PATH:$JAVA_HOME/bin
ENV HADOOP_VERSION=3.3.6
ENV HADOOP_HOME=/opt/hadoop
ENV SPARK_VERSION=3.5.1
ENV SPARK_HOME=/opt/spark

WORKDIR /opt
RUN wget -q https://archive.apache.org/dist/hadoop/common/hadoop-${HADOOP_VERSION}/hadoop-${HADOOP_VERSION}.tar.gz && \
    tar xzf hadoop-${HADOOP_VERSION}.tar.gz && \
    mv hadoop-${HADOOP_VERSION} hadoop && \
    rm hadoop-${HADOOP_VERSION}.tar.gz && \
    mkdir -p ${HADOOP_HOME}/etc/hadoop && \
    chmod -R 755 ${HADOOP_HOME}
RUN wget -q https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop3.tgz && \
    tar xzf spark-${SPARK_VERSION}-bin-hadoop3.tgz && \
    mv spark-${SPARK_VERSION}-bin-hadoop3 spark && \
    rm spark-${SPARK_VERSION}-bin-hadoop3.tgz && \
    mkdir -p ${SPARK_HOME}/spark-events && \
    chmod -R 755 ${SPARK_HOME}
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout=3600 -r requirements.txt
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
ENV PATH=${HADOOP_HOME}/bin:${HADOOP_HOME}/sbin:${SPARK_HOME}/sbin:${SPARK_HOME}/bin:${CUDA_HOME}/bin:$PATH
ENV PYSPARK_PYTHON=python3

RUN mkdir -p /hadoop/dfs/name /hadoop/dfs/data /hadoop/yarn/local /hadoop/yarn/logs /hadoop/tmp && \
    mkdir -p /var/log/supervisor /var/log/hadoop /var/run/supervisor && \
    mkdir -p ${SPARK_HOME}/spark-events && \
    chmod -R 755 /hadoop ${SPARK_HOME} /var/log/supervisor /var/log/hadoop /var/run/supervisor
COPY supervisor-config/supervisord.conf /etc/supervisor/conf.d/supervisord.conf
COPY spark-config/spark-defaults.conf ${SPARK_HOME}/conf/
COPY hadoop-config/core-site.xml ${HADOOP_HOME}/etc/hadoop/
COPY hadoop-config/hdfs-site.xml ${HADOOP_HOME}/etc/hadoop/
COPY hadoop-config/yarn-site.xml ${HADOOP_HOME}/etc/hadoop/
COPY hadoop-config/mapred-site.xml ${HADOOP_HOME}/etc/hadoop/

RUN chmod +x ${SPARK_HOME}/sbin/* && \
    chmod +x ${SPARK_HOME}/bin/* && \
    chmod +x ${HADOOP_HOME}/bin/* && \
    chmod +x ${HADOOP_HOME}/sbin/*

WORKDIR ${SPARK_HOME}
CMD ["supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]