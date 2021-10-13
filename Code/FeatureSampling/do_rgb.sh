echo Number out of 5 indicates which script has started, not ended

#python3 RGB/get_data_autoenc.py
#python3 RGB/get_data_depthnorm.py

#python3 RGB/get_data_gabor.py
echo 1/5
python3 RGB/get_data_gabor_depth.py

echo 2/5
python3 RGB/get_data_logGabor_3rot.py
echo 3/5
python3 RGB/get_data_logGabor_8rot.py
echo 4/5
python3 RGB/get_data_logGabor_depth_3rot.py
echo 5/5
python3 RGB/get_data_logGabor_depth_8rot.py
