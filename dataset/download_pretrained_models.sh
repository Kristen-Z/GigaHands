echo -e "Downloading pretrained models in `checkpoints` folder."
wget https://g-852369.56197.5898.data.globus.org/checkpoints.zip
rm -rf checkpoints

unzip checkpoints.zip
echo -e "Cleaning\n"
rm checkpoints.zip

echo -e "Downloading done!"
