#!/bin/sh
# Lists the tensors (name, datatype, format, shape, byte size) stored in SQLite .ckpt weight files,
# without any weight data, so their layout can be mapped for loading.
#
# Usage: scripts/dump_ckpt_index.sh OUT_DIR FILE.ckpt [FILE.ckpt ...]
# Writes OUT_DIR/<file>.txt per checkpoint. Needs only the sqlite3 that ships with macOS.
set -eu
out="$1"; shift
mkdir -p "$out"
for f in "$@"; do
  name=$(basename "$f")
  {
    echo "# $name ($(wc -c < "$f" | tr -d ' ') bytes)"
    echo "# tables: $(sqlite3 -readonly "$f" "select group_concat(name, ' ') from sqlite_master where type = 'table'")"
    echo "# name|type|format|datatype|dim (hex, 12 x int32 LE)|data bytes"
    sqlite3 -readonly "$f" "select name, type, format, datatype, hex(dim), length(data) from tensors order by name"
  } > "$out/$name.txt"
  echo "$out/$name.txt: $(($(wc -l < "$out/$name.txt") - 3)) tensors"
done
