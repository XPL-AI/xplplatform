#!/bin/bash
RESET_VENV=False
SKIP_REQUIREMENTS=False

XPL_HOST=$1
shift

if [[ "$XPL_HOST" != xpl* ]] ;
then
  XPL_HOST="xpl.$XPL_HOST"
fi

PROJECT_DIR="$XPL_CODE_DIR/${XPL_HOST//.//}";
VENV_DIR="$XPL_CODE_DIR/venvs/${XPL_HOST//.//}";

CUDA_VERSION=`nvidia-smi | egrep -o "CUDA Version: [[:digit:]]{1,2}.[[:digit:]]{1,2}" | sed "s/^.*: //"`

# # # Retrieving parameters
# when script is sourced getopts only works once
# if you don't reset: OPTIND=1
OPTIND=1
while getopts ":r:s" flag; do
  case ${flag} in
        r)
          RESET_VENV=True
          ;;
        s)
          SKIP_REQUIREMENTS=True
          ;;
        *)
          ;;
  esac
done
shift $((OPTIND -1))
echo "----------XPL_HOST:    $XPL_HOST";
echo "----------PROJECT_DIR: $PROJECT_DIR";
echo "----------VENV_DIR:    $VENV_DIR"
echo "----------RESET_VENV:  $RESET_VENV";
echo "----------CUDA_VERSION:  $CUDA_VERSION";


if [ ! -d "$PROJECT_DIR" ]
then
	echo "----ERR---PROJECT_DIR=$PROJECT_DIR does not exist."
	exit 1
fi


# Verify if virtual environment folder exists.
if [ -d "$VENV_DIR" ]
then
	echo "----------venv found: $VENV_DIR"
	if [ "$RESET_VENV" = True ]
	then
		echo "----------venv removing: $VENV_DIR"
		rm -rf "$VENV_DIR"
		echo "----------virtualenv --python=python3.9 $VENV_DIR"
		virtualenv --python=python3.9 "$VENV_DIR"
	fi
else
	echo "----------venv not found: $VENV_DIR"
	echo "----------virtualenv --python=python3.9 $VENV_DIR"
	virtualenv --python=python3.9 "$VENV_DIR"
fi

echo "----------$VENV_DIR/bin/activate"
. "$VENV_DIR/bin/activate"

export XPL_HOST=$XPL_HOST
export PYTHONPATH=$XPL_CODE_DIR

if [ "$SKIP_REQUIREMENTS" = False ]
then
  echo "----------pip3 install -r $PROJECT_DIR/requirements.txt -q"
  pip3 install -r "$PROJECT_DIR/requirements.txt" -q
  echo "----------pip3 wheel -r $PROJECT_DIR/requirements.txt --wheel-dir=$XPL_CODE_DIR/libs/${XPL_HOST//./-} --no-deps -q"
  pip3 wheel -r "$PROJECT_DIR/requirements.txt" --wheel-dir="$XPL_CODE_DIR/libs/${XPL_HOST//./-}" --no-deps -q
fi

`$XPL_CODE_DIR/bin/python -m visdom.server >/dev/null 2>&1 &`