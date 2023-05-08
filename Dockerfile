FROM ubuntu:latest

WORKDIR /usr/app/src

ARG LANG='en_us.UTF-8'

RUN apt-get update \ 
    && apt-get install -y --no-install-recommends \ 
        apt-utils \ 
        # build-essentials \ 
        locales \ 
        python3-pip \ 
        python3-yaml \ 
        rsyslog systemd systemd-cron sudo \ 
    && apt-get clean

RUN pip3 install --upgrade pip

RUN pip3 install streamlit

COPY / ./

CMD ['streamlit', 'run', 'Home.py']