#!/bin/bash
# lists the functions of a module whose facts say they allocate: probe/allocs.sh FILE
f=$1
dir=$(dirname $f); base=$(basename $f)
grep -oE "^fn (\([a-z]+ (mut )?([A-Za-z0-9_]+)(<[^>]*>)?&?\) )?[A-Za-z0-9_.]+" $f | sed -E 's/^fn \([a-z]+ (mut )?([A-Za-z0-9_]+)(<[^>]*>)?&?\) /\2./; s/^fn //; s/<.*//' | sort -u | while read name; do
  out=$(cd $dir && /home/user/olang-bin/bin/olang explain $base $name 2>&1 | grep "effects" | grep -m1 "allocates")
  [ -n "$out" ] && echo "$name: $out"
done
