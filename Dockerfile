FROM demisto/python3-deb:3.12.10.3561556

RUN apt-get update -y && \
    apt-get install -y \
        libx11-dev \
        python3-tk \
        git \
        libgl1 \
        x11-apps \
        libglib2.0-0 \
        curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install uv

COPY . /jemdzem
WORKDIR /jemdzem
RUN chmod +x entrypoint.sh
RUN uv run true

ENTRYPOINT ["/jemdzem/entrypoint.sh"]
