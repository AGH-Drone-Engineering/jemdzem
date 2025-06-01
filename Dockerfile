FROM demisto/python3-deb:3.12.10.3561556

RUN apt-get update -y
RUN apt-get install -y libx11-dev
RUN apt-get install -y python3-tk
RUN apt-get install -y git
RUN apt-get install -y libgl1
RUN apt-get install -y x11-apps
RUN apt-get install -y libglib2.0-0
RUN apt-get install -y curl

RUN pip install uv

RUN git clone https://github.com/AGH-Drone-Engineering/jemdzem.git

WORKDIR /jemdzem
RUN uv run true

COPY entrypoint.sh /jemdzem/entrypoint.sh
RUN chmod +x /jemdzem/entrypoint.sh

ENTRYPOINT ["/jemdzem/entrypoint.sh"]
