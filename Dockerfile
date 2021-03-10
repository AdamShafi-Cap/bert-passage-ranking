FROM python:3.8.5-buster
WORKDIR /workdir
COPY requirements.txt /workdir
COPY app_aws_ver.py /workdir
RUN mkdir /workdir/embeddings
RUN mkdir /workdir/models
COPY /embeddings/*.pkl /workdir/embeddings

EXPOSE 8501

RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get -y install python3-pip
RUN apt-get install git -y
RUN pip3 install -r requirements.txt 

RUN mkdir -p /root/.streamlit
RUN bash -c 'echo -e "\
[general]\n\
email = \"\"\n\
" > /root/.streamlit/credentials.toml'
RUN bash -c 'echo -e "\
[server]\n\
enableCORS = false\n\
" > /root/.streamlit/config.toml'

CMD ["streamlit", "run", "app_aws_ver.py"]