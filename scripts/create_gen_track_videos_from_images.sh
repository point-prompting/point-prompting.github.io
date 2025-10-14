#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

IN_ROOT="gen_videos"
OUT_ROOT="gen_videos_videos"
FPS=15

mkdir -p "$OUT_ROOT"

for dir in "$IN_ROOT"/*/; do
  folder_name="$(basename "$dir")"
  out="$OUT_ROOT/$folder_name.mp4"

  # Collect frames (full paths) and bail if none
  frames=( "$dir"frame_*.png )
  if [ ${#frames[@]} -eq 0 ]; then
    echo "No frames in: $dir (skipping)"
    continue
  fi

  # Extract numeric indices, sorted
  idx_list="$(printf '%s\n' "${frames[@]}" \
    | sed -E 's/.*frame_([0-9]+)\.png/\1/' \
    | sort -n)"

  start="$(echo "$idx_list" | head -n1)"
  end="$(echo "$idx_list" | tail -n1)"
  actual="$(echo "$idx_list" | wc -l | tr -d ' ')"
  expected=$(( end - start + 1 ))
  has_gap="$(awk 'NR>1 && $1!=prev+1{print "gap"; exit}{prev=$1}' <<< "$idx_list" || true)"

  echo "Building $out from $dir (frames $start..$end, count=$actual)"

  if [ "$has_gap" != "gap" ] && [ "$actual" -eq "$expected" ]; then
    # Contiguous sequence → simple, fast
    ffmpeg -y -framerate "$FPS" -start_number "$start" \
      -i "$dir"frame_%d.png \
      -vf "format=yuv420p" \
      "$out"
  else
    # Gapped sequence → concat list with natural sort
    listfile="$(mktemp)"
    printf '%s\n' "${frames[@]}" \
      | sort -V \
      | sed "s/^/file '/;s/\$/'/" > "$listfile"

    ffmpeg -y -r "$FPS" -f concat -safe 0 \
      -i "$listfile" \
      -vf "format=yuv420p" \
      "$out"

    rm -f "$listfile"
  fi
done