#ignorning depth, covered in rgb

# No script currently for autoencoder with LAB colour
#python3 LAB/get_data_autoenc.py

echo 0/3
python3 LAB/get_data_gabor.py

echo 1/3
python3 LAB/get_data_logGabor_3rot.py
echo 2/3
python3 LAB/get_data_logGabor_8rot.py
echo 3/3
