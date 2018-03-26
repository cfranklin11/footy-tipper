# Use an official Python runtime as a parent image
FROM python:3.4

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
ADD requirements.txt requirements.txt

# Install boost python for vowpal wabbit
RUN apt-get update && apt-get -y install postgresql-client \
  python3-dev
  #libboost-program-options-dev zlib1g-dev libboost-python-dev

# From https://askubuntu.com/a/363716
RUN cd /usr/local/include \
  && ln -s ../../include/python3.4 . \
  && cd /

# From https://github.com/lballabio/dockerfiles/blob/master/boost/Dockerfile
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y build-essential wget libbz2-dev \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

RUN wget https://dl.bintray.com/boostorg/release/1.65.0/source/boost_1_65_0.tar.gz \
  && tar xfz boost_1_65_0.tar.gz \
  && rm boost_1_65_0.tar.gz \
  && cd boost_1_65_0 \
  && ./bootstrap.sh --with-python=/usr/local/bin/python3 --with-python-version=3.4 --with-python-root=/usr/local/lib/python3.4 \
  && ./b2 \
  && ./b2 install \
  # && ./b2 --prefix=/usr -j 4 link=shared runtime-link=shared install \
  && cd .. && rm -rf boost_1_65_0 && ldconfig

# Install any needed packages specified in requirements.txt
RUN pip3 install --trusted-host pypi.python.org -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Run run.py when the container launches
CMD ["python3", "run.py"]
