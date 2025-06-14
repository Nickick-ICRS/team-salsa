# Team Salsa - LeRobot Global Hackathon 2025/06/14

## Installation and Setup

First, clone this repository, and make sure to include the submodules.
```
git clone --recurse-submodules -j8 git://github.com/Nickick-ICRS/team-salsa
cd team-salsa
```

Activate the docker environment provided
```
TODO
```

Create a conda environment, ensuring that you use Python version 3.10:
```
conda create -n salsa_env python=3.10
```

You may need to fix permissions:
```
sudo chown -R 1000:1000 ./
sudo chown -R 1000:1000 /path/to/conda/packages
```

Install dial-mpc:
```
cd dial-mpc && pip install -e .
```

Make sure dial-mpc works:
```
dial-mpc --list-examples
```

cd to our folder:
```
cd ../team-salsa
```