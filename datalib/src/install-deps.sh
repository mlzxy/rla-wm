cat <<EOF > /tmp/requirements.txt
pyspacemouse==2.0.0
torch==2.9.1
numpy==2.4.1
mani_skill==3.0.0b22
sapien==3.0.2
rich==14.3.1
opencv-python==4.11.0.86
transforms3d==0.4.2
EOF

conda install pinocchio -c conda-forge
python -m pip install -r /tmp/requirements.txt
brew install hidapi@0.15.0

echo "Notes: 

1. configure HIDAPI per https://spacemouse.kubaandrysek.cz/#examples
2. run 'pkill -f 3Dconnexion' to kill the 3Dconnexion process, if it exists.
   otherwise, pyspacemouse will fail to open the device.
"