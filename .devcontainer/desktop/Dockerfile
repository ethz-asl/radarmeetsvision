FROM nvcr.io/nvidia/pytorch:23.04-py3

ARG FLIRIMAGING_GID=1001
ARG TZ=Europe/Zurich
ARG VGLUSERS_GID=1004
ENV DEBIAN_FRONTEND=noninteractive
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/opt/hpcx/ompi/lib"
ENV PATH="${PATH}:/opt/hpcx/ompi/bin"

# All apt packages
COPY devcontainer_all_packages.sh /tmp/devcontainer_all_packages.sh
RUN /tmp/devcontainer_all_packages.sh && rm -f /tmp/devcontainer_all_packages.sh

ENV LANG="en_US.UTF-8" \
    LANGUAGE="en_US:en" \
    LC_ALL="en_US.UTF-8"

# Python packages
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --ignore-installed -r /tmp/requirements.txt && \
    rm -f /tmp/requirements.txt

# Add user "asl" with sudo rights
RUN groupadd -r asl && \
    groupadd -g ${FLIRIMAGING_GID} flirimaging && \
    groupadd -g ${VGLUSERS_GID} vglusers && \
    useradd --create-home --gid asl --groups dialout,plugdev,flirimaging,vglusers --shell /bin/bash asl && \
    mkdir -p /etc/sudoers.d && \
    echo 'asl ALL=NOPASSWD: ALL' > /etc/sudoers.d/asl

# ROS: noetic install
COPY install_ros_noetic_and_deps.sh /tmp/install_ros_noetic_and_deps.sh
RUN bash /tmp/install_ros_noetic_and_deps.sh

# Spinnaker: Install spinnaker SDK for flir cameras
COPY spinnaker/spinnaker-3.2.0.62-amd64 /tmp/spinnaker-3.2.0.62-amd64
WORKDIR /tmp/spinnaker-3.2.0.62-amd64
# Figured out by running: RUN echo $(debconf-show libgentl)
RUN echo "libspinnaker libspinnaker/accepted-flir-eula boolean true" | debconf-set-selections
RUN echo "libspinnaker libspinnaker/present-flir-eula boolean true" | debconf-set-selections
RUN /tmp/spinnaker-3.2.0.62-amd64/install_spinnaker.sh
RUN /tmp/spinnaker-3.2.0.62-amd64/configure_usbfs.sh

# Git-prompt: Source
RUN echo 'if [ -f "$HOME/.bash-git-prompt/gitprompt.sh" ]; then GIT_PROMPT_ONLY_IN_REPO=1; source "$HOME/.bash-git-prompt/gitprompt.sh"; fi' >> /home/asl/.bash_aliases

# TI: Download and install UniFlash
WORKDIR /tmp
RUN wget https://dr-download.ti.com/software-development/software-programming-tool/MD-QeJBJLj8gq/8.7.0/uniflash_sl.8.7.0.4818.run && \
    chmod +x uniflash_sl.8.7.0.4818.run && \
    ./uniflash_sl.8.7.0.4818.run --mode unattended --prefix /home/asl/ti/uniflash --debuglevel 0

# Clean up
RUN rm -rf /tmp/* /var/lib/apt/lists/* /var/tmp/* /var/cache/apt/archives/*

ENTRYPOINT ["/bin/bash"]
