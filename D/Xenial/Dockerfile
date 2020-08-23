
ARG BASE_IMAGE=rstudio/r-base
FROM ${BASE_IMAGE}:xenial

ARG R_VERSION=4.0.2
ARG OS_IDENTIFIER=ubuntu-1604

# Install R
RUN wget https://cdn.rstudio.com/r/${OS_IDENTIFIER}/pkgs/r-${R_VERSION}_1_amd64.deb && \
    apt-get update -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install -f -y ./r-${R_VERSION}_1_amd64.deb && \
    ln -s /opt/R/${R_VERSION}/bin/R /usr/bin/R && \
    ln -s /opt/R/${R_VERSION}/bin/Rscript /usr/bin/Rscript && \
    ln -s /opt/R/${R_VERSION}/lib/R /usr/lib/R && \
    rm r-${R_VERSION}_1_amd64.deb && \
    rm -rf /var/lib/apt/lists/*

# Install git
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git

# Clone the files into the docker container
RUN git clone https://github.com/henry090/fastai.git k

WORKDIR k

# Install dependecies
RUN R -e "install.packages('remotes', repos='http://cran.us.r-project.org')"
RUN R -e "remotes::install_deps(dependencies =  TRUE)"
RUN R -e "remotes::install_cran('rcmdcheck',repos='http://cran.us.r-project.org')"


# Install python
RUN R -e "remotes::install_github('rstudio/reticulate')"
RUN R -e "try(reticulate::install_miniconda())"
RUN R -e "reticulate::conda_create('r-reticulate', packages = 'python==3.6.10')"


# Install FASTAI
RUN R -e "remotes::install_local()"
RUN R -e "reticulate::py_install('torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html',pip = TRUE)"
RUN R -e "reticulate::py_install('fastai2==0.0.30',pip = TRUE)"
RUN R -e "reticulate::py_install(c('IPython'), pip = TRUE)"

# Check
RUN R -e "rcmdcheck::rcmdcheck(args = '--no-manual', error_on = 'error', check_dir = 'check')"

