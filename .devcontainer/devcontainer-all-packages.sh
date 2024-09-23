#!/bin/bash

set -eux
export DEBIAN_FRONTEND=noninteractive

main() {
    local pkgs=(
        apt-transport-https
        bison
        build-essential
        ca-certificates
        ccache
        curl
        gawk
        gnupg
        htop
        libgconf-2-4
        libnss3
        libusb-1.0-0
        locales
        net-tools
        python3-dev
        python3-pip
        python3-pyqt5
        screen
        software-properties-common
        ssh
        sudo
        udev
        unzip
        usbutils
        wget
    )

    apt-get update
    apt-get upgrade -y
    apt-get -y --quiet --no-install-recommends install "${pkgs[@]}"

    mkdir -p /root/.ssh \
        && chmod 0700 /root/.ssh \
        && ssh-keyscan github.com > /root/.ssh/known_hosts

    ln -snf /usr/share/zoneinfo/${TZ} /etc/localtime && echo ${TZ} > /etc/timezone
    sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen
    locale-gen en_US.UTF-8
    dpkg-reconfigure locales

    apt-get -y autoremove
    apt-get clean autoclean
    rm -rf /var/lib/apt/lists/{apt,dpkg,cache,log} /tmp/* /var/tmp/*
}

main "$@"
