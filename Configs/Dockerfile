FROM ubuntu
RUN apt-get update && apt-get install -y python3 && apt-get install -y python3-pip
RUN apt-get install -y vim
RUN apt-get install -y git
RUN mkdir /configfiles
COPY packages.txt /configfiles
RUN pip3 install -r /configfiles/packages.txt
RUN pip3 install git+https://github.com/vd1371/PyImbalReg.git
