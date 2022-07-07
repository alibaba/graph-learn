SCRIPT_DIR=$( cd "${0%/*}" && pwd -P )

PYTHON_SRC_FILES=$(find python -name '*.py')
if [[ -z ${PYTHON_SRC_FILES} ]]; then
  echo "Found no Python files to check."
fi
echo "${PYTHON_SRC_FILES}"

PYLINTRC_FILE="${SCRIPT_DIR}/pylintrc"
echo "${PYLINTRC_FILE}"
if [[ ! -f "${PYLINTRC_FILE}" ]]; then
  echo "ERROR: Cannot find pylint rc file at ${PYLINTRC_FILE}"
fi
echo "Running pylint..."
python3 -c "import pylint"
if [[ $?>0 ]] ; then
  pip3 install pylint -i https://pypi.tuna.tsinghua.edu.cn/simple \
  -trusted-host pypi.org --trusted-host pypi.python.org \
  --trusted-host files.pythonhosted.org  \
  --trusted-host pypi.tuna.tsinghua.edu.cn
fi
PYLINT_OUTPUT_FILE="$(mktemp)_pylint_output.log"
rm -rf "${PYLINT_OUTPUT_FILE}"
python3 -m pylint --rcfile="${PYLINTRC_FILE}" --output-format=parseable \
    "${PYTHON_SRC_FILES}" > "${PYLINT_OUTPUT_FILE}" 2>&1

echo "Pylint done. Result stored in ${PYLINT_OUTPUT_FILE}"

echo "Running cpplint..."
python3 -c "import cpplint"
if [[ $?>0 ]] ; then
  pip3 install cpplint -i https://pypi.tuna.tsinghua.edu.cn/simple \
  -trusted-host pypi.org --trusted-host pypi.python.org \
  --trusted-host files.pythonhosted.org  \
  --trusted-host pypi.tuna.tsinghua.edu.cn
fi
CPPLINT_OUTPUT_FILE="$(mktemp)_cpplint_output.log"
rm -rf "${CPPLINT_OUTPUT_FILE}"
CPPLINT_FILE="${SCRIPT_DIR}/cpplint.py"
python3 "${CPPLINT_FILE}" --recursive \
  --exclude=src/service/generated/*.cc \
  --exclude=src/service/test/test_helper.h \
  src/* > "${CPPLINT_OUTPUT_FILE}" 2>&1

echo "Cpplint done. Result stored in ${CPPLINT_OUTPUT_FILE}"
