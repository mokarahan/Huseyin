#!/usr/bin/bash

source $HOME/.venv/bin/activate
CMD=$1
shift

case "$CMD" in
    d)
        python main.py download
        ;;
    p)
        python main.py prepare
        ;;
    t)
        python main.py train --window=10 $@
        ;;

    *)
        echo "No command! "
	echo "d: download p:prepare t:train"
        ;;
esac



