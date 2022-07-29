#!/bin/bash

here=$(dirname "$(realpath "$0")")
fbs_bin_dir="${here}"/../../third_party/flatbuffers/build/bin
fbs_cpp_file_dir="${here}"/../fbs
fbs_java_file_dir="${here}"/fbs
fbs_gen_dir="${here}"/src/main/java

mkdir -p "${fbs_java_file_dir}"
rm -f "${fbs_java_file_dir}"/*.fbs

for file in "${fbs_cpp_file_dir}"/*.fbs
do
  if test -f "${file}"
  then
    file_name=$(basename "${file}")
    sed -e "s@namespace dgs@namespace org.aliyun.dgs@g" \
      "${file}" > "${fbs_java_file_dir}"/"${file_name}"
  fi
done

"${fbs_bin_dir}"/flatc -o "${fbs_gen_dir}" --java -I "${fbs_java_file_dir}" "${fbs_java_file_dir}"/*.fbs

rm -rf "${fbs_java_file_dir}"
