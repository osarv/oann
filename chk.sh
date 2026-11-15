#!/bin/sh
export OLANG_STD=/home/user/olang/std
for f in examples/*.olang bench/*.olang tools/*.olang repro/*.olang; do
  echo "== $f"
  /home/user/olang-bin/olang build -output build/chk_$(basename $f .olang) $f 2>&1 | tail -6
done
echo DONE
