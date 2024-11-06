for ((i=0; i<64; i+=1))
do
  start=$((i))
  end=$((start + 1))
  core=$((i % 80))
  nohup taskset -c $core /home/gaoyujie/OpenLRM/blender-3.6.0-linux-x64_0/blender -b -P /home/gaoyujie/OpenLRM/scripts/data/objaverse/blender_script.py -- \
      --object_path /data2/facescape/topologically_uniformed_model \
      --output_dir /home/wcc/lrm_data \
      --start $start \
      --end $end \
      --num_images 300 \
      > blender_"$i".log 2>&1 &
  echo "Blender instance started on core $core, processing range $start to $end."
done