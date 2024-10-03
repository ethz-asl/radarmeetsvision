FROM nvcr.io/nvidia/pytorch:22.12-py3

ARG TZ=Europe/Zurich
ENV DEBIAN_FRONTEND=noninteractive
ENV BLENDER_DIR=/home/asl/blender
ARG VGLUSERS_GID=1004

# Base packages
COPY devcontainer_all_packages.sh /tmp/devcontainer_all_packages.sh
RUN /tmp/devcontainer_all_packages.sh && rm -f /tmp/devcontainer_all_packages.sh

ENV LANG="en_US.UTF-8" \
    LANGUAGE="en_US:en" \
    LC_ALL="en_US.UTF-8" \
    BLENDER_USER_SCRIPTS=/home/asl/.config/blender/3.0/scripts/

# Add user "asl" with sudo rights
RUN groupadd -r asl && \
    groupadd -g ${VGLUSERS_GID} vglusers && \
    useradd --create-home --gid asl --groups dialout,plugdev,vglusers --shell /bin/bash asl && \
    mkdir -p /etc/sudoers.d && \
    echo 'asl ALL=NOPASSWD: ALL' > /etc/sudoers.d/asl

# Blender: Compile with patch that allows rendering depth in background
RUN apt-get --assume-yes install xpra subversion cmake libx11-dev libxxf86vm-dev libxcursor-dev libxi-dev libxrandr-dev libxinerama-dev libglew-dev
RUN mkdir -p $BLENDER_DIR/lib && cd $BLENDER_DIR/lib && svn checkout https://svn.blender.org/svnroot/bf-blender/tags/blender-3.0-release/lib/linux_centos7_x86_64/
RUN mkdir -p $BLENDER_DIR && cd $BLENDER_DIR && git clone https://github.com/marcojob/blender.git && cd blender && git checkout v3_0_1_background_patch && git submodule update --init --recursive
RUN cd $BLENDER_DIR/blender && make
RUN cd $BLENDER_DIR/blender && cmake ../build_linux -D WITH_CYCLES_CUDA_BINARIES=ON && make
RUN echo "alias blender='$BLENDER_DIR/build_linux/bin/blender'" >> /home/asl/.bash_aliases

# Blender: Install python deps
RUN $BLENDER_DIR/build_linux/bin/3.0/python/bin/python3.9 -m ensurepip
RUN $BLENDER_DIR/build_linux/bin/3.0/python/bin/python3.9 -m pip install pyyaml matplotlib scipy

# Blender: Copy user preferences to enable CUDA (ideally this would work with bpy, not figured out)
COPY userpref.blend /home/asl/.config/blender/3.0/config/

# Blender addon: Download and install BlenderGIS
RUN wget https://github.com/domlysz/BlenderGIS/archive/refs/tags/2210.zip -O /tmp/BlenderGIS-2210.zip && \
    unzip /tmp/BlenderGIS-2210.zip -d /tmp/ && \
    mkdir -p $BLENDER_USER_SCRIPTS/addons && \
    mv /tmp/BlenderGIS-2210 $BLENDER_USER_SCRIPTS/addons/BlenderGIS && \
    $BLENDER_DIR/build_linux/bin/blender -b --python-expr "import bpy; bpy.ops.preferences.addon_enable(module='BlenderGIS'); bpy.ops.wm.save_userpref()"

# Git-prompt: Source
RUN echo 'if [ -f "$HOME/.bash-git-prompt/gitprompt.sh" ]; then GIT_PROMPT_ONLY_IN_REPO=1; source "$HOME/.bash-git-prompt/gitprompt.sh"; fi' >> /home/asl/.bash_aliases

# Clean up
RUN rm -rf /tmp/* /var/lib/apt/lists/* /var/tmp/* /var/cache/apt/archives/*

ENTRYPOINT ["/bin/bash"]
