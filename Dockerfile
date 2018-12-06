FROM btel/nrnpython:nrn7.4py2.7

MAINTAINER telenczuk@unic.cnrs-gif.fr 


RUN pip install --user git+https://github.com/maikia/neuroneap
RUN pip install --user numpy matplotlib==1.5.3 scipy

# Make sure the contents of our repo are in ${HOME}
COPY . ${HOME}
USER root
RUN chown -R ${NB_UID} ${HOME}
USER ${NB_USER}
