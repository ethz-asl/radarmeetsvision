#!/bin/bash

if [ ! -d "$HOME/.bash-git-prompt" ]; then
    mkdir "$HOME/.bash-git-prompt"
fi

if [ ! -d "$HOME/.cache" ]; then
    mkdir "$HOME/.cache"
fi

if [ ! -d "$HOME/Storage" ]; then
    mkdir "$HOME/Storage"
fi
