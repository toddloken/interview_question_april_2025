FROM continuumio/miniconda3

WORKDIR /app

COPY . /app

RUN conda env create -f conda_environment.yml
RUN echo "source activate housing" > ~/.bashrc
ENV PATH /opt/conda/envs/housing/bin:$PATH

CMD ["python", "app.py"]
