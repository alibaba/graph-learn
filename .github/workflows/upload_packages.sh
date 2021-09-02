#!/bin/bash
set -e

if [[ "$GITHUB_REF" =~ ^"refs/tags/" ]]; then
  export GITHUB_TAG_REF="$GITHUB_REF"
  export GIT_TAG=$(echo "$GITHUB_REF" | sed -e "s/refs\/tags\///g")
fi

if [ -z "$GITHUB_TAG_REF" ]; then
  echo "Not on a tag, won't deploy to pypi"
else
  docker pull $DOCKER_IMAGE
  pyabis=$(echo $PYABI | tr ":" "\n")
  for abi in $pyabis; do
    docker run --rm -e "PYABI=$abi" -e "GIT_TAG=$GIT_TAG" -v `pwd`:/io \
      $DOCKER_IMAGE $PRE_CMD bash -c "chmod +x /io/.github/workflows/build.sh; /io/.github/workflows/build.sh"
    sudo chown -R $(id -u):$(id -g) ./*
    mv dist/*.whl /tmp
  done
  mv /tmp/*.whl dist/

  echo "********"
  echo "Build packages:"
  ls dist/
  echo "********"

  echo "[distutils]"                                 > ~/.pypirc
  echo "index-servers ="                             >> ~/.pypirc
  echo "    pypi"                                    >> ~/.pypirc
  echo "[pypi]"                                      >> ~/.pypirc
  echo "repository=https://upload.pypi.org/legacy/"  >> ~/.pypirc
  echo "username=__token__"                             >> ~/.pypirc
  echo "password=$PYPI_PWD"                          >> ~/.pypirc

  python -m pip install twine
  python -m twine upload -r pypi --skip-existing dist/*
fi