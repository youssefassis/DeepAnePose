FROM nvidia/cuda:11.6.0-base-ubuntu20.04
FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

# edit these
ENV NB_USER username
ENV GROUP groupname
ENV PYTHONPATH="${PYTHONPATH}:/path/to/poseaneurysm/dataStrategy:/path/to/poseaneurysm/models:/path/to/poseaneurysm/ressources"

ENV HOME /home/${NB_USER}

WORKDIR ${HOME}
COPY ./requirements.txt requirements.txt
RUN DEBIAN_FRONTEND=noninteractive apt-get update --fix-missing && \
    apt-get install -y git libglib2.0-0 libsm6 libxrender1 libxext6 build-essential && \
    apt-get clean                 && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --upgrade pip && \
    pip install -r requirements.txt  && \
    pip install hydra-core --upgrade --pre && \
    addgroup ${GROUP}  && \
    adduser --ingroup ${GROUP} --home ${HOME} --disabled-password --shell /bin/bash --gecos "" ${NB_USER}  && \
    adduser ${NB_USER} sudo && \
    jupyter notebook --generate-config
   
COPY jupyter_notebook_config.json ${HOME}/.jupyter/jupyter_notebook_config.json
RUN chown -R ${NB_USER}:${GROUP} ${HOME}/.jupyter
USER ${NB_USER}

EXPOSE 5000
VOLUME ${HOME}
CMD ["jupyter", "lab"]
