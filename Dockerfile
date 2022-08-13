FROM python:3.8

COPY requirements.txt /usr/src/app/requirements.txt
WORKDIR /usr/src/app

RUN pip3 install -r requirements.txt

COPY . /usr/src/app

EXPOSE 8000